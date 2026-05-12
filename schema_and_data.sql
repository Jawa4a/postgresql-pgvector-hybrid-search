/*
    1. aktiveeri pgvector;
    2. loo tabelid;
    3. muuda COPY käskude failitee;
    4. laadi andmed CSV-failidest tabelitesse.
*/

CREATE EXTENSION IF NOT EXISTS vector;

-- Kui vaja tabelid uuesti luua, siis jäta järgmised read kommenteerimata:
-- DROP TABLE IF EXISTS public.news_articles_10k;
-- DROP TABLE IF EXISTS public.news_articles_25k;
-- DROP TABLE IF EXISTS public.news_articles_50k;

CREATE TABLE IF NOT EXISTS public.news_articles_10k (
    id BIGINT PRIMARY KEY,
    category TEXT NOT NULL,
    headline TEXT NOT NULL,
    short_description TEXT NOT NULL,
    authors TEXT,
    published_date DATE NOT NULL,
    link TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);

CREATE TABLE IF NOT EXISTS public.news_articles_25k (
    id BIGINT PRIMARY KEY,
    category TEXT NOT NULL,
    headline TEXT NOT NULL,
    short_description TEXT NOT NULL,
    authors TEXT,
    published_date DATE NOT NULL,
    link TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);

/*
    Enne COPY käskude käivitamist asenda failiteed.
    Iga tee peab viitama vastava alamhulga failile news_with_embeddings_pgvector.csv.
*/

CREATE TABLE IF NOT EXISTS public.news_articles_50k (
    id BIGINT PRIMARY KEY,
    category TEXT NOT NULL,
    headline TEXT NOT NULL,
    short_description TEXT NOT NULL,
    authors TEXT,
    published_date DATE NOT NULL,
    link TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);

COPY public.news_articles_10k (
    id,
    category,
    headline,
    short_description,
    authors,
    published_date,
    link,
    content,
    embedding
    )
    FROM 'C:\...\prepared_data\subset_10000\news_with_embeddings_pgvector.csv'
    WITH (
    FORMAT csv,
    HEADER true,
    ENCODING 'UTF8',
    QUOTE '"',
    ESCAPE '"'
);

COPY public.news_articles_25k (
    id,
    category,
    headline,
    short_description,
    authors,
    published_date,
    link,
    content,
    embedding
    )
    FROM 'C:\...\prepared_data\subset_25000\news_with_embeddings_pgvector.csv'
    WITH (
    FORMAT csv,
    HEADER true,
    ENCODING 'UTF8',
    QUOTE '"',
    ESCAPE '"'
);

COPY public.news_articles_50k (
    id,
    category,
    headline,
    short_description,
    authors,
    published_date,
    link,
    content,
    embedding
    )
    FROM 'C:\...\prepared_data\subset_50000\news_with_embeddings_pgvector.csv'
    WITH (
    FORMAT csv,
    HEADER true,
    ENCODING 'UTF8',
    QUOTE '"',
    ESCAPE '"'
);