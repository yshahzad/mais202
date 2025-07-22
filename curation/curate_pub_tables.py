import pandas as pd
import numpy as np
from config import NUM_AUTHORS_RECORDED, CATEGORY_LABELS


#Tested and works
def curate_pub_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Curate a publications table from a DataFrame.
    Returns a DataFrame with base info, split authors, and exploded categories.
    """
    
    #Parse Dates
    df['created_date'] = pd.to_datetime(df['latest_created_date']).dt.date
    df['created_month'] = pd.to_datetime(df['created_date']).dt.to_period('M')

    #Exploded categories df
    categories_df = df['categories'].str.split(' ').explode()

    #Authors df
    authors_df = df['authors'].str.replace(' and ', ', ').str.split(', ', expand=True, n= NUM_AUTHORS_RECORDED - 1)

    #Write columns dictionary in list comprehension format


    authors_df = authors_df.rename(
    columns = {i: f'auth_{i+1}' for i in range(NUM_AUTHORS_RECORDED)}
        )
    authors_df['num_authors'] = df['authors'].str.replace(' and ', ', ').str.split(', ').str.len()

    #Merge tables
    #DO NOT reset index as the merge logic relies on the index
    base_df = df[['id', 'title', 'comments', 'created_date', 'created_month']]
    tables = [base_df, authors_df, categories_df]
    return pd.concat(tables, axis=1)

def get_unique_authors_by_month_and_category(discipline_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (month, category) combination, compute the number of unique authors 
    (up to different spellings, initialisms, etc.).
    """
    author_cols = [f'auth_{i + 1}' for i in range(NUM_AUTHORS_RECORDED)]

    melted_df = discipline_df.melt(
        id_vars=['created_month', 'categories'],
        value_vars=author_cols,
        value_name='author'
    ).dropna(subset=['author'])

    unique_author_counts = (
    melted_df
    .groupby(['created_month', 'categories'])['author']
    .nunique()
    .reset_index(name='num_unique_authors')
    )

    return unique_author_counts


def get_all_pubs_by_month(discipline_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each month, compute the number of publications across all categories for a given discipline.
    """
    pub_counts = discipline_df.groupby('created_month').size().reset_index(name='total_monthly_pubs')
    return pub_counts


def aggregate_pub_table(curated_df: pd.DataFrame) -> pd.DataFrame:
    pub_count_df = curated_df.groupby(['created_month', 'categories'])['created_month'].count()
    pub_count_df = pub_count_df.reset_index(name='num_pubs')
    return pub_count_df


def features_pub_table(curated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a features DataFrame with all relevant covariates for publications.
    """

    pub_counts_df = aggregate_pub_table(curated_df)


    unique_auths = get_unique_authors_by_month_and_category(curated_df)

    features_df = pd.merge(pub_counts_df, unique_auths, on=['created_month', 'categories'], how='inner')
    features_df = pd.merge(features_df, get_all_pubs_by_month(curated_df), on='created_month', how='inner')

    features_df['cat_name'] = features_df['categories'].apply(lambda x: CATEGORY_LABELS.get(x, 'Unknown'))

    return features_df

