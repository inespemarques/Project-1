#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import re
import numpy as np
from sklearn.decomposition import PCA

# Function to read the first line and extract samples
def extract_samples_from_first_line(file_path):
    df = pd.read_csv(file_path, header=None, nrows=1, skip_blank_lines=False)
    first_line = df.iloc[0, 0]
    samples = first_line.strip('"').split('","')
    return samples

# Function to extract gene names
def extract_gene_names(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)
    pattern = re.compile(r'ENSG[^,]*')
    gene_names = [pattern.search(row).group(0) for row in df[0] if pattern.search(row)]
    return gene_names

# Function to extract numbers from each row after the gene
def extract_numbers_from_rows(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)
    all_numbers = []
    pattern = re.compile(r'ENSG[^,]*')
    for index, row in df.iterrows():
        row_str = ','.join(row.astype(str))
        numbers_str = re.sub(pattern, '', row_str).strip(',')
        numbers = [float(num) for num in numbers_str.split(',') if num]
        all_numbers.append(numbers)
    return all_numbers

# Function to remove the lowest variance genes
def remove_low_variance_genes(file_path, top_n=4000):
    all_numbers = extract_numbers_from_rows(file_path)
    genes = extract_gene_names(file_path)
    variances = [np.var(numbers) for numbers in all_numbers]
    gene_variance_pairs = list(zip(genes, variances))
    sorted_genes = sorted(gene_variance_pairs, key=lambda x: x[1])
    top_genes = set(gene for gene, _ in sorted_genes[:top_n])
    
    filtered_genes = [gene for gene in genes if gene not in top_genes]
    filtered_numbers = [numbers for gene, numbers in zip(genes, all_numbers) if gene not in top_genes]
    
    return filtered_genes, filtered_numbers

# Function to apply PCA and filter genes based on relevant PCA coefficients
def apply_pca_and_filter_genes(filtered_numbers, filtered_genes, n_components=2, top_n_genes=2000):
    pca = PCA(n_components=n_components)
    pca.fit(filtered_numbers)
    
    components = pca.components_
    relevant_genes = set()
    
    for i in range(n_components):
        component = components[i]
        gene_contributions = sorted(zip(filtered_genes, component), key=lambda x: abs(x[1]), reverse=True)
        top_genes = [gene for gene, _ in gene_contributions[:top_n_genes]]
        relevant_genes.update(top_genes)

    # Filter to keep only the relevant genes
    filtered_genes_pca = [gene for gene in filtered_genes if gene in relevant_genes]
    filtered_numbers_pca = [numbers for gene, numbers in zip(filtered_genes, filtered_numbers) if gene in relevant_genes]
    
    return filtered_genes_pca, filtered_numbers_pca


file_path = 'dataset1.csv'
filtered_genes, filtered_numbers = remove_low_variance_genes(file_path, top_n=4000)
filtered_genes_pca, filtered_numbers_pca = apply_pca_and_filter_genes(filtered_numbers, filtered_genes)

# Display results
print(f"Filtered Genes (PCA): {filtered_genes_pca[:5]}")
print(f"Filtered Vectors (PCA): {filtered_numbers_pca[:5]}")


# In[12]:


len(filtered_genes)


# In[10]:


filtered_genes_pca


# In[ ]:




