import os
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import math
import io
import argparse
from torch.nn.utils.rnn import pad_sequence
from deap import base, creator, tools, algorithms
from multiprocessing import Pool
import random
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "sshleifer/tiny-gpt2"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.85  
MIN_CHUNK_SIZE = 64  
MAX_CHUNK_SIZE = 1024  
CONTEXT_WINDOW = 256  
BATCH_SIZE = 16  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POPULATION_SIZE = 25  
GENERATIONS = 40  
ELITISM_RATE = 0.3  
MUTATION_RATE = 0.5
DELETION_RATE = 0.6
CROSSOVER_RATE = 0.8
NUM_WORKERS = 4 
TOURNAMENT_SIZE = 6
MAX_MUTATIONS_PER_INDIVIDUAL = 5  
TOKEN_IMPORTANCE_THRESHOLD = 0.3

# logging configuration
def setup_logging(chunk_id):
    """Setup detailed logging for a chunk"""
    log_dir = "debug_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"chunk_{chunk_id}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

class Text8Dataset(Dataset):
    """Dataset for text with configurable chunking"""
    def __init__(self, file_path, max_seq_length=MAX_CHUNK_SIZE, chunk_size=None):
        self.file_path = file_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_length = max_seq_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # specified input size
        if chunk_size is not None:
            text = text[:chunk_size]
        
        # Tokenize the entire text with truncation
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_length).input_ids[0]
        
        # Split tokens into chunks of max_seq_length
        self.chunks = []
        for i in range(0, len(tokens), max_seq_length):
            chunk = tokens[i:i+max_seq_length]
            self.chunks.append(chunk)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

# Initialize Sentence Transformer for semantic similarity
sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL).to(DEVICE)

def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts using Sentence Transformers"""
    # Encode texts to get embeddings
    embeddings1 = sentence_model.encode(text1, convert_to_tensor=True).to(DEVICE)
    embeddings2 = sentence_model.encode(text2, convert_to_tensor=True).to(DEVICE)
    
    # calculate cosine similarity
    similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=0).item()
    return similarity

def compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of text using the model"""
    # Tokenize text
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity

def evaluate_token_importance(model, tokenizer, tokens, position):
    """Evaluate the importance of a token at a given position using the LLM"""
    if position >= len(tokens) or position < 0:
        return 0.0
    
    # Get context before and after the token
    context_before = tokens[:position]
    token_to_evaluate = tokens[position:position+1]
    context_after = tokens[position+1:position+CONTEXT_WINDOW+1]
    
    # Calculate perplexity with the token
    with torch.no_grad():
        # Full context
        full_context = torch.cat([context_before, token_to_evaluate, context_after])
        if len(full_context) > MAX_CHUNK_SIZE:
            full_context = full_context[-MAX_CHUNK_SIZE:]
        
        # Compute loss with the token
        if len(full_context) > 1:
            outputs_with = model(full_context.unsqueeze(0), labels=full_context.unsqueeze(0))
            loss_with = outputs_with.loss
        else:
            return 0.5
        
        # Context without the token
        context_without = torch.cat([context_before, context_after])
        if len(context_without) > MAX_CHUNK_SIZE:
            context_without = context_without[-MAX_CHUNK_SIZE:]
        
        if len(context_without) > 1:
            outputs_without = model(context_without.unsqueeze(0), labels=context_without.unsqueeze(0))
            loss_without = outputs_without.loss
        else:
            return 1.0
    
    # Calculate importance based on how much the loss increases when removing the token
    if loss_with.item() < 1e-10:
        importance = 1.0
    else:
        # Fixed: Ensure float division
        importance = max(0.0, min(1.0, float((loss_without - loss_with).item() / loss_with.item())))
    
    return importance

def llmzip_with_arithmetic_coding(model, tokenizer, tokens: torch.Tensor) -> float:
    """LLMZip implementation with Arithmetic Coding"""
    if len(tokens) <= 1:
        return 0.0
    
    compressed_size_bits = 0.0
    
    # Process each token with sliding window context
    for i in range(1, len(tokens)):
        # Get context window
        start_idx = max(0, i - CONTEXT_WINDOW)
        context = tokens[start_idx:i]
        
        # Ensure context doesn't exceed model's maximum capacity
        if len(context) > MAX_CHUNK_SIZE:
            context = context[-MAX_CHUNK_SIZE:]
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(context.unsqueeze(0))
            logits = outputs.logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
            
            # Get the actual token
            actual_token = tokens[i].item()
            
            # Calculate probability
            prob = probs[actual_token].item()
            
            # Avoid log(0)
            if prob < 1e-10:
                prob = 1e-10
            
            # Add to compressed size
            compressed_size_bits += -math.log2(prob)
    
    return compressed_size_bits

def calculate_compression_gain(original_size: float, alternative_size: float) -> float:
    """Calculate compression gain"""
    if original_size == 0:
        return 0.0
    return (original_size - alternative_size) / original_size

def calculate_bpc(compressed_size_bits: float, text: str) -> float:
    """Calculate bits per character"""
    num_chars = len(text)
    if num_chars == 0:
        return 0.0
    return compressed_size_bits / num_chars

# Fitness function 
def calculate_fitness(individual, original_chunk, original_compressed_size, model, tokenizer):
    """Calculate fitness based on supervisor's formula: fitness(s') = (LLMZip(s) - LLMZip(s'))/LLMZip(s) if sim(s,s')>=theta else 0"""
    # Convert individual to text
    individual_text = tokenizer.decode(individual.tensor)
    
    # Get original text
    original_text = tokenizer.decode(original_chunk)
    
    # Compute semantic similarity
    similarity = compute_semantic_similarity(original_text, individual_text)
    
    # Always calculate compressed size for BPC calculation
    alternative_compressed_size = llmzip_with_arithmetic_coding(model, tokenizer, individual.tensor)
    
    # Calculate BPC
    bpc = calculate_bpc(alternative_compressed_size, individual_text)
    
    # Calculate compression gain: (LLMZip(s) - LLMZip(s')) / LLMZip(s)
    compression_gain = calculate_compression_gain(original_compressed_size, alternative_compressed_size)
    
    # fitness condition
    if similarity >= SEMANTIC_THRESHOLD:
        fitness = compression_gain
    else:
        fitness = 0.0
    
    return fitness, similarity, alternative_compressed_size, bpc

# Define Individual class
class Individual:
    def __init__(self, tensor):
        self.tensor = tensor
        self.fitness = creator.FitnessMax()
        self.id = id(self)
        self.similarity = 0.0  # Store similarity for selection

# Setup DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
toolbox = base.Toolbox()

def initialize_individual(original_chunk, model, tokenizer):
    """Initialize an individual with small mutations"""
    # Start with a copy of the original
    tensor = original_chunk.clone()
    
    # Apply a small number of mutations for diversity
    num_mutations = min(5, max(1, int(0.1 * len(tensor)))) 
    
    mutation_positions = torch.randperm(len(tensor))[:num_mutations]
    
    for pos in mutation_positions:
        # Get token importance
        importance = evaluate_token_importance(model, tokenizer, tensor, pos.item())
        
        # Only mutate less important tokens
        if importance < TOKEN_IMPORTANCE_THRESHOLD:
            # Get context for prediction
            context = tensor[:pos] if pos > 0 else tensor[:1]
            
            # Skip if context is empty
            if len(context) == 0:
                continue
            
            # Get model predictions
            with torch.no_grad():
                if len(context) > MAX_CHUNK_SIZE:
                    context = context[-MAX_CHUNK_SIZE:]
                
                outputs = model(context.unsqueeze(0))
                logits = outputs.logits[0, -1]
                probs = torch.softmax(logits, dim=-1)
                
                # For less important tokens, consider substitution
                if torch.rand(1).item() < 0.7:  # Increased chance of substitution
                    # Get top tokens that would result in lower bits
                    original_token = tensor[pos].item()
                    token_probs = []
                    
                    # Check top 10 tokens
                    top_tokens = torch.topk(probs, 10)
                    for token in top_tokens.indices.tolist():
                        if token != original_token:
                            prob = probs[token].item()
                            if prob < 1e-10:
                                prob = 1e-10
                            bits = -math.log2(prob)
                            token_probs.append((token, bits))
                    
                    # Sort by bits (lower is better)
                    token_probs.sort(key=lambda x: x[1])
                    
                    if token_probs:
                        # Choose from the top 3 tokens
                        best_tokens = token_probs[:3]
                        new_token = best_tokens[torch.randint(0, len(best_tokens), (1,)).item()][0]
                        tensor[pos] = new_token
    
    return Individual(tensor)

def llm_guided_mutation(individual, model, tokenizer, mutation_rate: float = MUTATION_RATE, generation: int = 0):
    """LLM-guided mutation with gradual changes"""
    if torch.rand(1).item() > mutation_rate or len(individual.tensor) == 0:
        return individual, False, "none"
    
    # Calculate current similarity
    original_text = tokenizer.decode(individual.tensor)
    similarity = compute_semantic_similarity(original_text, tokenizer.decode(individual.tensor))
    
    # Adjust mutation aggressiveness based on similarity
    if similarity < SEMANTIC_THRESHOLD:
        # Be more conservative when similarity is low
        max_mutations = min(3, max(1, int(0.05 * len(individual.tensor))))
        mutation_rate_adjusted = mutation_rate * 0.7
    else:
        max_mutations = min(MAX_MUTATIONS_PER_INDIVIDUAL, 3 + generation // 10)
        mutation_rate_adjusted = mutation_rate
    
    if torch.rand(1).item() > mutation_rate_adjusted:
        return individual, False, "none"
    
    # Select random positions for mutation
    num_mutations = min(max_mutations, max(1, int(0.1 * len(individual.tensor))))  # Increased mutations
    mutation_positions = torch.randperm(len(individual.tensor))[:num_mutations]
    
    mutated_tensor = individual.tensor.clone()
    mutation_occurred = False
    mutation_type = "none"
    
    # Convert to list for easier manipulation
    positions_list = mutation_positions.tolist()
    
    for i, pos in enumerate(positions_list):
        # Check if position is still valid after previous deletions
        if pos >= len(mutated_tensor):
            continue
            
        # Get token importance
        importance = evaluate_token_importance(model, tokenizer, mutated_tensor, pos)
        
        # Decide mutation type based on importance and random chance
        if importance < TOKEN_IMPORTANCE_THRESHOLD and len(mutated_tensor) > MIN_CHUNK_SIZE:
            # Less important token - consider deletion or substitution
            if torch.rand(1).item() < DELETION_RATE:
                # Delete token at this position
                mutated_tensor = torch.cat([mutated_tensor[:pos], mutated_tensor[pos+1:]])
                mutation_occurred = True
                mutation_type = "deletion"
                
                # Adjust remaining positions after deletion
                for j in range(i+1, len(positions_list)):
                    if positions_list[j] > pos:
                        positions_list[j] -= 1
            else:
                # Substitute with a token that requires fewer bits
                context = mutated_tensor[:pos] if pos > 0 else mutated_tensor[:1]
                
                if len(context) > 0:
                    with torch.no_grad():
                        if len(context) > MAX_CHUNK_SIZE:
                            context = context[-MAX_CHUNK_SIZE:]
                        
                        outputs = model(context.unsqueeze(0))
                        logits = outputs.logits[0, -1]
                        probs = torch.softmax(logits, dim=-1)
                        
                        original_token = mutated_tensor[pos].item()
                        original_bits = -math.log2(max(1e-10, probs[original_token].item()))
                        
                        # Find tokens that require fewer bits
                        better_tokens = []
                        for token, prob in enumerate(probs):
                            if token != original_token:
                                bits = -math.log2(max(1e-10, prob.item()))
                                if bits < original_bits:
                                    better_tokens.append((token, bits))
                        
                        if better_tokens:
                            # Sort by bits (lower is better)
                            better_tokens.sort(key=lambda x: x[1])
                            # Choose from the best options
                            new_token = better_tokens[torch.randint(0, min(3, len(better_tokens)), (1,)).item()][0]
                            mutated_tensor[pos] = new_token
                            mutation_occurred = True
                            mutation_type = "substitution"
    
    if mutation_occurred:
        mutated_individual = Individual(mutated_tensor)
        # Invalidate fitness
        del mutated_individual.fitness.values
        return mutated_individual, True, mutation_type
    else:
        return individual, False, "none"

def similarity_preserving_crossover(parent1, parent2, model, tokenizer):
    """Conservative crossover that preserves important tokens"""
    if len(parent1.tensor) < 2 or len(parent2.tensor) < 2:
        return parent1, parent2, False
    
    # Find crossover points where both parents have low importance tokens
    min_len = min(len(parent1.tensor), len(parent2.tensor))
    
    # Calculate token importance for both parents
    importance1 = []
    for i in range(min_len):
        importance = evaluate_token_importance(model, tokenizer, parent1.tensor, i)
        importance1.append(importance)
    
    importance2 = []
    for i in range(min_len):
        importance = evaluate_token_importance(model, tokenizer, parent2.tensor, i)
        importance2.append(importance)
    
    # Combined importance (lower is better for crossover)
    combined_importance = torch.tensor(importance1, dtype=torch.float32) + torch.tensor(importance2, dtype=torch.float32)
    
    # Find positions with low combined importance
    low_importance_threshold = torch.quantile(combined_importance, 0.15).item()  # More aggressive
    candidate_positions = torch.where(combined_importance < low_importance_threshold)[0] + 1
    
    # Filter out positions too close to edges
    if len(candidate_positions) > 0:
        edge_buffer = max(1, int(0.15 * min_len))  
        candidate_positions = candidate_positions[
            (candidate_positions >= edge_buffer) & 
            (candidate_positions <= min_len - edge_buffer)
        ]
    
    if len(candidate_positions) == 0 or torch.rand(1).item() > CROSSOVER_RATE:
        return parent1, parent2, False
    
    # Select a random crossover point from candidates
    crossover_point = candidate_positions[torch.randint(0, len(candidate_positions), (1,)).item()]
    
    # Create children
    child1_tensor = torch.cat([parent1.tensor[:crossover_point], parent2.tensor[crossover_point:]])
    child2_tensor = torch.cat([parent2.tensor[:crossover_point], parent1.tensor[crossover_point:]])
    
    child1 = Individual(child1_tensor)
    child2 = Individual(child2_tensor)
    # Invalidate fitness
    del child1.fitness.values
    del child2.fitness.values
    
    return child1, child2, True

def evaluate_individual(individual, original_chunk, original_compressed_size, model, tokenizer):
    """Evaluate an individual's fitness"""
    fitness, similarity, compressed_size, bpc = calculate_fitness(
        individual, original_chunk, original_compressed_size, model, tokenizer
    )
    
    # Store similarity in individual for selection
    individual.similarity = similarity
    
    # Calculate perplexity
    perplexity = compute_perplexity(model, tokenizer, tokenizer.decode(individual.tensor))
    
    # Set fitness values
    individual.fitness.values = (fitness,)
    
    return (fitness,), {
        "similarity": similarity,
        "compressed_size": compressed_size,
        "bpc": bpc,
        "perplexity": perplexity,
        "text_length": len(tokenizer.decode(individual.tensor))
    }

def tournament_selection(population, tournament_size=TOURNAMENT_SIZE):
    """Tournament selection that prioritizes similarity"""
    selected = []
    
    for _ in range(len(population)):
        # Select tournament_size individuals at random
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_individuals = [population[i] for i in tournament_indices]
        
        # First filter: Only consider individuals with similarity above threshold
        valid_candidates = [ind for ind in tournament_individuals if hasattr(ind, 'similarity') and ind.similarity >= SEMANTIC_THRESHOLD]
        
        if valid_candidates:
            # If we have valid candidates, select the best among them
            best = max(valid_candidates, key=lambda ind: ind.fitness.values[0])
        else:
            # Otherwise, select the one with highest similarity
            best = max(tournament_individuals, key=lambda ind: getattr(ind, 'similarity', 0))
        
        selected.append(best)
    
    return selected

def calculate_diversity(population):
    """Calculate population diversity"""
    if len(population) <= 1:
        return 0.0
    
    diversity = 0.0
    count = 0
    
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            # Calculate Hamming distance between tensors
            len1 = len(population[i].tensor)
            len2 = len(population[j].tensor)
            min_len = min(len1, len2)
            
            # Pad shorter tensor with zeros
            tensor1 = torch.nn.functional.pad(population[i].tensor, (0, max(0, len2 - len1)))
            tensor2 = torch.nn.functional.pad(population[j].tensor, (0, max(0, len1 - len2)))
            
            # Calculate Hamming distance
            hamming_dist = torch.sum(tensor1 != tensor2).item() / max(len1, len2)
            diversity += hamming_dist
            count += 1
    
    return diversity / count if count > 0 else 0.0

def post_process_for_similarity(model, tokenizer, original_text, alternative_text, threshold=SEMANTIC_THRESHOLD):
    """Post-process alternative text to ensure it meets similarity threshold"""
    current_similarity = compute_semantic_similarity(original_text, alternative_text)
    
    if current_similarity >= threshold:
        return alternative_text
    
    # If below threshold, try to restore important tokens
    original_tokens = tokenizer.encode(original_text, return_tensors="pt")[0].to(DEVICE)
    alternative_tokens = tokenizer.encode(alternative_text, return_tensors="pt")[0].to(DEVICE)
    
    # Find tokens that were removed or changed
    # Try to restore some tokens that might be important for semantics
    restored_tokens = alternative_tokens.clone()
    
    # Add some original tokens back at key positions
    # This is a placeholder for a more sophisticated restoration algorithm
    for i in range(min(15, len(original_tokens))):  
        if i < len(restored_tokens):
            # Restore some tokens at regular intervals
            if i % 3 == 0:  # More frequent restoration
                restored_tokens[i] = original_tokens[i]
    
    restored_text = tokenizer.decode(restored_tokens)
    restored_similarity = compute_semantic_similarity(original_text, restored_text)
    
    # If restoration improved similarity, use it
    if restored_similarity > current_similarity:
        return restored_text
    
    # If all else fails, return the original text
    return original_text

def process_chunk_with_debugging(args):
    """Process a single chunk with genetic algorithm and detailed debugging"""
    chunk, model, tokenizer, chunk_id = args
    
    # Setup logging for this chunk
    log_file = setup_logging(chunk_id)
    logging.info(f"Processing chunk {chunk_id} with {len(chunk)} tokens")
    
    # Compress original chunk using LLMZip with arithmetic coding
    original_compressed_size = llmzip_with_arithmetic_coding(model, tokenizer, chunk)
    original_text = tokenizer.decode(chunk)
    original_bpc = calculate_bpc(original_compressed_size, original_text)
    original_perplexity = compute_perplexity(model, tokenizer, original_text)
    
    logging.info(f"Original text: {original_text[:100]}...")
    logging.info(f"Original compressed size: {original_compressed_size:.2f} bits")
    logging.info(f"Original BPC: {original_bpc:.4f}")
    logging.info(f"Original perplexity: {original_perplexity:.2f}")
    
    # Setup DEAP
    toolbox.register("individual", initialize_individual, chunk, model, tokenizer)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", similarity_preserving_crossover, model=model, tokenizer=tokenizer)
    toolbox.register("mutate", llm_guided_mutation, model=model, tokenizer=tokenizer)
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", evaluate_individual, 
                    original_chunk=chunk, 
                    original_compressed_size=original_compressed_size, 
                    model=model, 
                    tokenizer=tokenizer)
    
    # Initialize population
    population = toolbox.population(n=POPULATION_SIZE)
    
    # Evaluate initial population
    fitnesses = []
    metrics_list = []
    for ind in population:
        fitness, metrics = toolbox.evaluate(ind)
        fitnesses.append(fitness[0])
        metrics_list.append(metrics)
    
    # Log initial population details
    logging.info("Initial Population:")
    for i, (ind, fitness, metrics) in enumerate(zip(population, fitnesses, metrics_list)):
        logging.info(f"Individual {i}: Fitness={fitness:.4f}, Similarity={metrics['similarity']:.4f}, "
                    f"BPC={metrics['bpc']:.4f}, Compressed Size={metrics['compressed_size']:.2f}, "
                    f"Perplexity={metrics['perplexity']:.2f}, Length={metrics['text_length']}")
    
    # Track best individual over generations
    best_overall = None
    best_fitness = -float('inf')
    best_bpc = float('inf')
    fitness_history = []
    bpc_history = []
    diversity_history = []
    
    # Generation statistics
    gen_stats = []
    
    # Run genetic algorithm 
    for gen in range(GENERATIONS):
        logging.info(f"\n=== Generation {gen+1}/{GENERATIONS} ===")
        
        # Select elite individuals
        elite_size = int(ELITISM_RATE * POPULATION_SIZE)
        elite = tools.selBest(population, elite_size)
        
        # Select offspring for reproduction from entire population
        offspring = toolbox.select(population, POPULATION_SIZE - elite_size)
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and track operations
        crossover_count = 0
        for i in range(0, len(offspring)-1, 2):
            child1 = offspring[i]
            child2 = offspring[i+1]
            if random.random() < CROSSOVER_RATE:
                child1, child2, crossover_occurred = toolbox.mate(child1, child2)
                if crossover_occurred:
                    crossover_count += 1
                    offspring[i] = child1
                    offspring[i+1] = child2
        
        # Apply mutation and track operations
        mutation_count = 0
        mutation_types = {"deletion": 0, "substitution": 0, "none": 0}
        for i in range(len(offspring)):
            mutant = offspring[i]
            # Pass generation to mutation function
            mutant, mutation_occurred, mutation_type = toolbox.mutate(mutant, generation=gen)
            if mutation_occurred:
                mutation_count += 1
                mutation_types[mutation_type] += 1
                offspring[i] = mutant
        
        logging.info(f"Crossover operations: {crossover_count}")
        logging.info(f"Mutation operations: {mutation_count} (Deletions: {mutation_types['deletion']}, Substitutions: {mutation_types['substitution']})")
        
        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        new_fitnesses = []
        new_metrics = []
        
        logging.info(f"Evaluating {len(invalid_ind)} new individuals...")
        for ind in invalid_ind:
            fitness, metrics = toolbox.evaluate(ind)
            new_fitnesses.append(fitness[0])
            new_metrics.append(metrics)
        
        # Log new individuals
        for i, (ind, fitness, metrics) in enumerate(zip(invalid_ind, new_fitnesses, new_metrics)):
            logging.info(f"New Individual {i}: Fitness={fitness:.4f}, Similarity={metrics['similarity']:.4f}, "
                        f"BPC={metrics['bpc']:.4f}, Compressed Size={metrics['compressed_size']:.2f}, "
                        f"Perplexity={metrics['perplexity']:.2f}, Length={metrics['text_length']}")
        
        # Form next generation: elite + offspring
        population[:] = elite + offspring
        
        # Calculate generation statistics
        current_fitnesses = [ind.fitness.values[0] for ind in population]
        current_metrics = []
        
        for ind in population:
            _, metrics = toolbox.evaluate(ind)
            current_metrics.append(metrics)
        
        avg_fitness = np.mean(current_fitnesses)
        max_fitness = np.max(current_fitnesses)
        min_fitness = np.min(current_fitnesses)
        std_fitness = np.std(current_fitnesses)
        
        avg_similarity = np.mean([m['similarity'] for m in current_metrics])
        avg_bpc = np.mean([m['bpc'] for m in current_metrics])
        min_bpc = np.min([m['bpc'] for m in current_metrics])
        avg_perplexity = np.mean([m['perplexity'] for m in current_metrics])
        
        # Calculate diversity
        diversity = calculate_diversity(population)
        
        gen_stat = {
            "generation": gen+1,
            "avg_fitness": avg_fitness,
            "max_fitness": max_fitness,
            "min_fitness": min_fitness,
            "std_fitness": std_fitness,
            "avg_similarity": avg_similarity,
            "avg_bpc": avg_bpc,
            "min_bpc": min_bpc,
            "avg_perplexity": avg_perplexity,
            "diversity": diversity,
            "crossover_count": crossover_count,
            "mutation_count": mutation_count,
            "deletion_count": mutation_types["deletion"],
            "substitution_count": mutation_types["substitution"]
        }
        gen_stats.append(gen_stat)
        
        logging.info(f"Generation Stats: Avg Fitness={avg_fitness:.4f}, Max Fitness={max_fitness:.4f}, "
                    f"Min Fitness={min_fitness:.4f}, Std Fitness={std_fitness:.4f}")
        logging.info(f"                 Avg Similarity={avg_similarity:.4f}, Avg BPC={avg_bpc:.4f}, "
                    f"Min BPC={min_bpc:.4f}, Avg Perplexity={avg_perplexity:.2f}")
        logging.info(f"                 Diversity={diversity:.4f}")
        
        # Track best individual
        current_best = tools.selBest(population, 1)[0]
        current_fitness = current_best.fitness.values[0]
        
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_overall = current_best
        
        # Track best BPC
        current_bpc = min_bpc
        if current_bpc < best_bpc:
            best_bpc = current_bpc
        
        fitness_history.append(current_fitness)
        bpc_history.append(avg_bpc)
        diversity_history.append(diversity)
        
    
    # If no individual met the threshold, try to find the one with highest similarity
    if best_overall is None or best_overall.similarity < SEMANTIC_THRESHOLD:
        logging.info("No individual met threshold. Finding individual with highest similarity...")
        best_similarity = 0
        for ind in population:
            if ind.similarity > best_similarity:
                best_similarity = ind.similarity
                best_overall = ind
        logging.info(f"Best similarity: {best_similarity:.4f}")
    
    # Get text for the best individual
    best_text = tokenizer.decode(best_overall.tensor)
    
    # Post-process to ensure similarity threshold is met
    best_text = post_process_for_similarity(model, tokenizer, original_text, best_text)
    
    # Calculate statistics
    original_size = len(original_text.encode('utf-8'))
    alternative_size = len(best_text.encode('utf-8'))
    similarity = compute_semantic_similarity(original_text, best_text)
    perplexity = compute_perplexity(model, tokenizer, best_text)
    
    # Get final metrics
    _, best_metrics = toolbox.evaluate(best_overall)
    
    result = {
        "text": best_text,
        "original_size": original_size,
        "alternative_size": alternative_size,
        "similarity": similarity,
        "perplexity": perplexity,
        "fitness": best_overall.fitness.values[0],
        "compressed_size": best_metrics["compressed_size"],
        "bpc": best_metrics["bpc"],
        "generation_stats": gen_stats,
        "log_file": log_file
    }
    
    # Save generation statistics to JSON
    stats_file = log_file.replace('.log', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(gen_stats, f, indent=2)
    
    # Plot BPC, fitness, and diversity over generations
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot([stat["generation"] for stat in gen_stats], [stat["avg_bpc"] for stat in gen_stats], 'b-', label='Avg BPC')
    plt.plot([stat["generation"] for stat in gen_stats], [stat["min_bpc"] for stat in gen_stats], 'r-', label='Min BPC')
    plt.axhline(y=original_bpc, color='g', linestyle='--', label='Original BPC')
    plt.xlabel('Generation')
    plt.ylabel('BPC')
    plt.title('BPC Over Generations')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot([stat["generation"] for stat in gen_stats], [stat["avg_fitness"] for stat in gen_stats], 'b-', label='Avg Fitness')
    plt.plot([stat["generation"] for stat in gen_stats], [stat["max_fitness"] for stat in gen_stats], 'r-', label='Max Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Generations')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([stat["generation"] for stat in gen_stats], [stat["diversity"] for stat in gen_stats], 'g-', label='Diversity')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Population Diversity')
    plt.legend()
    
    plt.tight_layout()
    plot_file = log_file.replace('.log', '_plot.png')
    plt.savefig(plot_file)
    plt.close()
    
    logging.info(f"Chunk processing complete. Best fitness: {best_fitness:.4f}, Best BPC: {best_bpc:.4f}")
    logging.info(f"Original size: {original_size} bytes, Alternative size: {alternative_size} bytes")
    logging.info(f"Similarity: {similarity:.4f}, Perplexity: {perplexity:.2f}")
    logging.info(f"Generation statistics saved to {stats_file}")
    logging.info(f"Convergence plot saved to {plot_file}")
    
    return result

def process_dataset(model, tokenizer, data_loader) -> Tuple[List[str], Dict[str, Any]]:
    """Process the entire dataset with genetic algorithm compression"""
    model.eval()
    
    alternative_chunks = []
    all_stats = {
        "original_sizes": [],
        "alternative_sizes": [],
        "similarities": [],
        "perplexities": [],
        "fitness_scores": [],
        "bpc_scores": []
    }
    
    # Prepare chunks for multiprocessing
    chunk_args = []
    chunk_id = 0
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Preparing chunks")):
        batch = batch.to(DEVICE)
        
        # Process each sequence in the batch
        for seq_idx, seq in enumerate(batch):
            # Remove padding tokens (0)
            seq = seq[seq != 0]
            
            print(f"\nPreparing chunk {batch_idx * BATCH_SIZE + seq_idx + 1} (length: {len(seq)})")
            
            # Use the sequence directly without semantic chunking
            chunk_args.append((seq, model, tokenizer, chunk_id))
            chunk_id += 1
    
    # Process chunks in parallel
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_chunk_with_debugging, chunk_args), total=len(chunk_args), desc="Processing chunks"))
    
    # Process results
    for result in results:
        alternative_chunks.append(result["text"])
        all_stats["original_sizes"].append(result["original_size"])
        all_stats["alternative_sizes"].append(result["alternative_size"])
        all_stats["similarities"].append(result["similarity"])
        all_stats["perplexities"].append(result["perplexity"])
        all_stats["fitness_scores"].append(result["fitness"])
        all_stats["bpc_scores"].append(result["bpc"])
    
    return alternative_chunks, all_stats

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LLMZip Lossy Text Compression')
    parser.add_argument('input_file', type=str, help='Input text file path')
    parser.add_argument('chunk_size', type=int, help='Size of text to process in bytes')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist.")
        return
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # Create dataset and dataloader
    print(f"Creating dataset from {args.input_file} with chunk size {args.chunk_size} bytes...")
    dataset = Text8Dataset(args.input_file, max_seq_length=MAX_CHUNK_SIZE, chunk_size=args.chunk_size)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Process dataset
    print("Starting genetic algorithm compression...")
    start_time = time.time()
    alternative_chunks, all_stats = process_dataset(model, tokenizer, dataloader)
    processing_time = time.time() - start_time
    
    # Combine alternative chunks
    alternative_text = "".join(alternative_chunks)
    
    # Save alternative text
    output_file = f"alternative_{os.path.basename(args.input_file)}"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(alternative_text)
    print(f"Alternative text saved to {output_file}")
    
    # Load original text for comparison
    with open(args.input_file, "r", encoding="utf-8") as f:
        original_text = f.read(args.chunk_size)  # Only compare the processed chunk
    
    # Calculate overall statistics
    original_size = len(original_text.encode('utf-8'))
    alternative_size = len(alternative_text.encode('utf-8'))
    
    # Compress both texts with LLMZip with arithmetic coding for final comparison
    print("\nCompressing original text...")
    original_tokens = tokenizer(original_text, return_tensors="pt", truncation=True, max_length=MAX_CHUNK_SIZE).input_ids[0].to(DEVICE)
    original_compressed_size = llmzip_with_arithmetic_coding(model, tokenizer, original_tokens)
    
    print("Compressing alternative text...")
    alternative_tokens = tokenizer(alternative_text, return_tensors="pt", truncation=True, max_length=MAX_CHUNK_SIZE).input_ids[0].to(DEVICE)
    alternative_compressed_size = llmzip_with_arithmetic_coding(model, tokenizer, alternative_tokens)
    
    # Calculate BPC
    original_bpc = calculate_bpc(original_compressed_size, original_text)
    alternative_bpc = calculate_bpc(alternative_compressed_size, alternative_text)
    
    # Calculate overall semantic similarity
    overall_similarity = compute_semantic_similarity(original_text, alternative_text)
    
    # Calculate compression gain
    compression_gain = calculate_compression_gain(original_compressed_size, alternative_compressed_size)
    
    # Calculate perplexity for alternative text
    alternative_perplexity = compute_perplexity(model, tokenizer, alternative_text)
    
    # Print results
    print("\n" + "="*80)
    print("COMPRESSION RESULTS")
    print("="*80)
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Original text size: {original_size} bytes")
    print(f"Alternative text size: {alternative_size} bytes")
    print(f"Raw size reduction: {original_size - alternative_size} bytes ({(1 - alternative_size/original_size)*100:.2f}%)")
    print()
    print(f"Original LLMZip compressed size: {original_compressed_size:.2f} bits")
    print(f"Alternative LLMZip compressed size: {alternative_compressed_size:.2f} bits")
    print(f"Compressed size reduction: {original_compressed_size - alternative_compressed_size:.2f} bits ({(1 - alternative_compressed_size/original_compressed_size)*100:.2f}%)")
    print(f"Compression gain: {compression_gain:.4f}")
    print()
    print(f"Original BPC: {original_bpc:.4f}")
    print(f"Alternative BPC: {alternative_bpc:.4f}")
    print(f"BPC improvement: {original_bpc - alternative_bpc:.4f}")
    print()
    print(f"Overall semantic similarity: {overall_similarity:.4f}")
    print(f"Similarity threshold: {SEMANTIC_THRESHOLD}")
    print(f"Threshold satisfied: {'Yes' if overall_similarity >= SEMANTIC_THRESHOLD else 'No'}")
    print(f"Alternative text perplexity: {alternative_perplexity:.2f}")
    print()
    print("PER-CHUNK STATISTICS")
    print("="*80)
    print(f"Number of chunks processed: {len(all_stats['original_sizes'])}")
    print(f"Average similarity: {np.mean(all_stats['similarities']):.4f}")
    print(f"Min similarity: {np.min(all_stats['similarities']):.4f}")
    print(f"Max similarity: {np.max(all_stats['similarities']):.4f}")
    print(f"Chunks below threshold: {sum(1 for s in all_stats['similarities'] if s < SEMANTIC_THRESHOLD)}")
    print()
    print(f"Average original size: {np.mean(all_stats['original_sizes']):.2f} bytes")
    print(f"Average alternative size: {np.mean(all_stats['alternative_sizes']):.2f} bytes")
    print(f"Average size reduction: {np.mean([o - a for o, a in zip(all_stats['original_sizes'], all_stats['alternative_sizes'])]):.2f} bytes")
    print()
    print(f"Average BPC: {np.mean(all_stats['bpc_scores']):.4f}")
    print(f"Min BPC: {np.min(all_stats['bpc_scores']):.4f}")
    print(f"Max BPC: {np.max(all_stats['bpc_scores']):.4f}")
    print()
    print(f"Average perplexity: {np.mean(all_stats['perplexities']):.2f}")
    print(f"Max perplexity: {np.max(all_stats['perplexities']):.2f}")
    print()
    print(f"Average fitness score: {np.mean(all_stats['fitness_scores']):.4f}")
    print(f"Max fitness score: {np.max(all_stats['fitness_scores']):.4f}")
    
    # Save detailed statistics
    stats_file = f"compression_stats_{os.path.basename(args.input_file)}.txt"
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("COMPRESSION STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Input file: {args.input_file}\n")
        f.write(f"Chunk size: {args.chunk_size} bytes\n")
        f.write(f"Processing time: {processing_time:.2f} seconds\n")
        f.write(f"Original text size: {original_size} bytes\n")
        f.write(f"Alternative text size: {alternative_size} bytes\n")
        f.write(f"Raw size reduction: {original_size - alternative_size} bytes ({(1 - alternative_size/original_size)*100:.2f}%)\n")
        f.write(f"\nOriginal LLMZip compressed size: {original_compressed_size:.2f} bits\n")
        f.write(f"Alternative LLMZip compressed size: {alternative_compressed_size:.2f} bits\n")
        f.write(f"Compressed size reduction: {original_compressed_size - alternative_compressed_size:.2f} bits ({(1 - alternative_compressed_size/original_compressed_size)*100:.2f}%)\n")
        f.write(f"Compression gain: {compression_gain:.4f}\n")
        f.write(f"\nOriginal BPC: {original_bpc:.4f}\n")
        f.write(f"Alternative BPC: {alternative_bpc:.4f}\n")
        f.write(f"BPC improvement: {original_bpc - alternative_bpc:.4f}\n")
        f.write(f"\nOverall semantic similarity: {overall_similarity:.4f}\n")
        f.write(f"Similarity threshold: {SEMANTIC_THRESHOLD}\n")
        f.write(f"Threshold satisfied: {'Yes' if overall_similarity >= SEMANTIC_THRESHOLD else 'No'}\n")
        f.write(f"Alternative text perplexity: {alternative_perplexity:.2f}\n")
        f.write("\nPER-CHUNK STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Number of chunks processed: {len(all_stats['original_sizes'])}\n")
        f.write(f"Average similarity: {np.mean(all_stats['similarities']):.4f}\n")
        f.write(f"Min similarity: {np.min(all_stats['similarities']):.4f}\n")
        f.write(f"Max similarity: {np.max(all_stats['similarities']):.4f}\n")
        f.write(f"Chunks below threshold: {sum(1 for s in all_stats['similarities'] if s < SEMANTIC_THRESHOLD)}\n")
        f.write(f"\nAverage original size: {np.mean(all_stats['original_sizes']):.2f} bytes\n")
        f.write(f"Average alternative size: {np.mean(all_stats['alternative_sizes']):.2f} bytes\n")
        f.write(f"Average size reduction: {np.mean([o - a for o, a in zip(all_stats['original_sizes'], all_stats['alternative_sizes'])]):.2f} bytes\n")
        f.write(f"\nAverage BPC: {np.mean(all_stats['bpc_scores']):.4f}\n")
        f.write(f"Min BPC: {np.min(all_stats['bpc_scores']):.4f}\n")
        f.write(f"Max BPC: {np.max(all_stats['bpc_scores']):.4f}\n")
        f.write(f"\nAverage perplexity: {np.mean(all_stats['perplexities']):.2f}\n")
        f.write(f"Max perplexity: {np.max(all_stats['perplexities']):.2f}\n")
        f.write(f"\nAverage fitness score: {np.mean(all_stats['fitness_scores']):.4f}\n")
        f.write(f"Max fitness score: {np.max(all_stats['fitness_scores']):.4f}\n")
        
        # Write per-chunk details
        f.write("\nPER-CHUNK DETAILS\n")
        f.write("="*80 + "\n")
        f.write("Chunk\tOrig Size\tAlt Size\tSimilarity\tBPC\tPerplexity\tFitness\n")
        for i in range(len(all_stats['original_sizes'])):
            f.write(f"{i+1}\t")
            f.write(f"{all_stats['original_sizes'][i]}\t")
            f.write(f"{all_stats['alternative_sizes'][i]}\t")
            f.write(f"{all_stats['similarities'][i]:.4f}\t")
            f.write(f"{all_stats['bpc_scores'][i]:.4f}\t")
            f.write(f"{all_stats['perplexities'][i]:.2f}\t")
            f.write(f"{all_stats['fitness_scores'][i]:.4f}\n")
    
    print(f"\nDetailed statistics saved to {stats_file}")
    print(f"Debug logs saved to debug_logs/ directory")

if __name__ == "__main__":
    main()