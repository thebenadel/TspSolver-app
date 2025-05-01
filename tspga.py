import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# ===================== Genetic Algorithm Functions ===================== #
def generate_initial_population(pop_size, n, start_node):
    """Generate random permutations starting with start_node"""
    return [[start_node] + random.sample([x for x in range(n) if x != start_node], n-1) 
            for _ in range(pop_size)]

def order_crossover(parent1, parent2):
    """Order Crossover (OX) for TSP"""
    n = len(parent1)
    start, end = sorted(random.sample(range(1, n-1), 2))
    offspring = parent1[start:end+1]
    
    current_pos = (end + 1) % n
    for gene in parent2:
        if gene not in offspring:
            if current_pos >= n:
                current_pos = 0
            offspring.insert(current_pos, gene)
            current_pos += 1
    return offspring[:n]

def mutate(path, mutation_rate):
    """Swap mutation"""
    path = path.copy()
    for i in range(1, len(path)-1):
        if random.random() < mutation_rate:
            j = random.randint(1, len(path)-2)
            path[i], path[j] = path[j], path[i]
    return path

def fitness(path, dist_matrix):
    """Calculate total path distance"""
    return sum(dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1)) + dist_matrix[path[-1]][path[0]]

def tournament_selection(population, dist_matrix, tournament_size=3):
    """Tournament parent selection"""
    candidates = random.sample(population, tournament_size)
    return min(candidates, key=lambda x: fitness(x, dist_matrix))

def genetic_algorithm(dist_matrix, start_node, generations=500, pop_size=100, mutation_rate=0.1):
    """Optimized GA implementation"""
    n = len(dist_matrix)
    population = generate_initial_population(pop_size, n, start_node)
    best_path = min(population, key=lambda x: fitness(x, dist_matrix))
    best_fitness = fitness(best_path, dist_matrix)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for gen in range(generations):
        # Update progress
        progress = (gen + 1) / generations
        progress_bar.progress(progress)
        status_text.text(f"Generation {gen+1}/{generations} | Current Best: {best_fitness:.1f}")
        
        # Breed new population
        new_population = []
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, dist_matrix)
            parent2 = tournament_selection(population, dist_matrix)
            child = mutate(order_crossover(parent1, parent2), mutation_rate)
            new_population.append(child)
        
        # Keep best solution
        population = sorted(population + new_population, key=lambda x: fitness(x, dist_matrix))[:pop_size]
        current_best = population[0]
        current_fitness = fitness(current_best, dist_matrix)
        
        if current_fitness < best_fitness:
            best_path = current_best
            best_fitness = current_fitness
    
    progress_bar.empty()
    status_text.empty()
    return best_path, best_fitness

# ===================== Streamlit App ===================== #
st.set_page_config(layout="wide")

# Initialize session state
if 'num_vertices' not in st.session_state:
    st.session_state.num_vertices = 5
if 'dist_matrix' not in st.session_state:
    st.session_state.dist_matrix = np.zeros((5, 5))
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'best_path' not in st.session_state:
    st.session_state.best_path = None
if 'best_distance' not in st.session_state:
    st.session_state.best_distance = None

# Sidebar controls
st.sidebar.header("TSP Configuration")

# Vertex count selector
num_vertices = st.sidebar.slider("Number of Vertices", 2, 10, 5)
if num_vertices != st.session_state.num_vertices:
    st.session_state.num_vertices = num_vertices
    st.session_state.dist_matrix = np.zeros((num_vertices, num_vertices))
    st.session_state.graph = None
    st.session_state.best_path = None

# Matrix input with automatic parsing
st.sidebar.subheader("Distance Matrix Input")
matrix_input = st.sidebar.text_area(
    f"Edit matrix ({num_vertices}x{num_vertices}):",
    value="[\n" + ",\n".join(["    " + str(row) for row in st.session_state.dist_matrix.tolist()]) + "\n]",
    height=300,
    key="matrix_editor"
)

# Auto-update matrix when edited
try:
    new_matrix = eval(matrix_input)
    if isinstance(new_matrix, list) and len(new_matrix) == num_vertices and all(len(row) == num_vertices for row in new_matrix):
        if all(new_matrix[i][i] == 0 for i in range(num_vertices)):
            if not np.array_equal(st.session_state.dist_matrix, new_matrix):
                st.session_state.dist_matrix = np.array(new_matrix, dtype=float)
                st.session_state.graph = None
                st.session_state.best_path = None
        else:
            st.sidebar.error("Diagonal elements must be 0")
    else:
        st.sidebar.error(f"Matrix must be {num_vertices}x{num_vertices}")
except Exception as e:
    st.sidebar.error(f"Invalid matrix format: {e}")

# Graph generation
if st.sidebar.button("Generate Graph"):
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            weight = st.session_state.dist_matrix[i][j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)
    st.session_state.graph = G
    st.session_state.best_path = None
    st.sidebar.success("Graph generated!")

# GA parameters
st.sidebar.subheader("Algorithm Parameters")
generations = st.sidebar.slider("Generations", 100, 2000, 500)
pop_size = st.sidebar.slider("Population Size", 50, 500, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
start_node = st.sidebar.selectbox("Start Node", options=list(range(num_vertices)), format_func=lambda x: f"Node {x}")

# Main interface
st.header("Traveling Salesman Problem Solver")

# Visualization
if st.session_state.graph:
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(st.session_state.graph)
    
    # Draw base graph
    nx.draw_networkx_nodes(st.session_state.graph, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(st.session_state.graph, pos)
    nx.draw_networkx_edges(st.session_state.graph, pos, edge_color='gray', alpha=0.3)
    
    # Draw best path if available
    if st.session_state.best_path:
        path_edges = list(zip(st.session_state.best_path, st.session_state.best_path[1:] + [st.session_state.best_path[0]]))
        nx.draw_networkx_edges(st.session_state.graph, pos, edgelist=path_edges, 
                              edge_color='red', width=2)
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(st.session_state.graph, 'weight')
    nx.draw_networkx_edge_labels(st.session_state.graph, pos, edge_labels=edge_labels)
    
    plt.title("TSP Graph Visualization")
    st.pyplot(fig)
else:
    st.info("Generate graph to see visualization")

# Controls and results
if st.button("Run Genetic Algorithm"):
    if st.session_state.graph is None:
        st.error("Please generate graph first!")
    else:
        with st.spinner("Optimizing TSP path..."):
            best_path, best_distance = genetic_algorithm(
                st.session_state.dist_matrix,
                start_node=start_node,
                generations=generations,
                pop_size=pop_size,
                mutation_rate=mutation_rate
            )
            
            st.session_state.best_path = best_path
            st.session_state.best_distance = best_distance
            st.rerun()

if st.session_state.best_path is not None:
    st.subheader("Results")
    st.write(f"**Best Path**: {' → '.join(map(str, st.session_state.best_path))} → {st.session_state.best_path[0]}")
    st.write(f"**Total Distance**: {st.session_state.best_distance:.2f}")

# Matrix validation info
st.sidebar.markdown("""
**Matrix Requirements**
- NxN numeric matrix (N = selected vertex count)
- Diagonal elements must be 0
- All other elements must be positive
- Matrix should be symmetric
""")