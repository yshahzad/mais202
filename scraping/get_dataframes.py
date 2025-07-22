import pandas as pd
import polars as pl
from tqdm import tqdm
import time
from arxiv_categories import physics_categories, math_categories, cs_categories, astro_categories

JSON_FILEPATH = "arxiv-metadata-oai-snapshot.json"
CSV_FILEPATH = "complete_arxiv_data.csv"

def JSON_to_CSV():
    # Specify the fields (tags) you want to keep
    fields_to_keep = ['id', 'authors', 'title', 'categories', 'comments', 'update_date', 'versions']

    # Open the file and read lines to get the total count
    with open(JSON_FILEPATH, 'r') as file:
        total_lines = sum(1 for line in file)

    # Load the JSON file in chunks and process with tqdm for a progress bar
    df_list = []
    with pd.read_json(JSON_FILEPATH, lines=True, chunksize=100000) as reader:
        for chunk in tqdm(reader, total=total_lines // 100000, desc="Processing JSON file"):
            filtered_chunk = pd.DataFrame({
                'id': chunk['id'],
                'authors': chunk['authors'],
                'title': chunk['title'],
                'categories': chunk['categories'],
                'comments': chunk.get('comments', pd.NA),
                'update_date': chunk.get('update_date', pd.NA),
                'latest_created_date': chunk['versions'].apply(
                    lambda x: x[-1]['created'] if isinstance(x, list) and len(x) > 0 else pd.NA
                )
            })
            df_list.append(filtered_chunk)
    
    # Concatenate all chunks
    result = pd.concat(df_list)
    result.to_csv(CSV_FILEPATH, index=False)



def get_discipline_entries(discipline, csv_filepath):
    print(f"Working on discipline: {discipline}")
    
    # Define category lists for each discipline
    discipline_categories = {
        'phys': physics_categories,
        'math': math_categories,
        'cs': cs_categories,
        'astro': astro_categories
    }

    if discipline not in discipline_categories:
        raise ValueError(f"Discipline '{discipline}' not recognized. Choose from 'phys', 'math', 'cs', or 'astro'.")

    # Define the df schema
    schema = {
        "id": pl.Utf8,
        "authors": pl.Utf8,
        "title": pl.Utf8,
        "categories": pl.Utf8,
        "comments": pl.Utf8,
        "update_date": pl.Utf8,
        "latest_created_date": pl.Utf8
    }
    
    # Load the CSV file with schema
    df = pl.read_csv(csv_filepath, schema=schema, row_index_name="index")

    df = df.with_columns(
    pl.col("latest_created_date").str.strptime(pl.Datetime, format="%a, %d %b %Y %H:%M:%S %Z", strict=False))

    categories = discipline_categories[discipline]
    filtered_df = df.filter(
        pl.col("categories").str.contains("|".join(categories))
    )

    print(filtered_df.head())

    # Save filtered data to a new CSV file
    filtered_csv_path = f"{discipline}_arxiv_data.csv"
    filtered_df.write_csv(filtered_csv_path)

    print(f"Filtered data saved to {filtered_csv_path}")



start_time = time.time()

# Run the functions

#JSON_to_CSV()
for discipline in ["math","cs", "phys", "astro"]:
    get_discipline_entries(discipline, "complete_arxiv_data.csv")

end_time = time.time()
print(f"\nCompleted in {(end_time - start_time) / 60:.2f} minutes.")


