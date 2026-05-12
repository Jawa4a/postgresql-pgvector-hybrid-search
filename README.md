# PostgreSQL + pgvector hübriidotsingu skriptid

See repositoorium sisaldab bakalaureusetöö praktilises osas kasutatud Python-skripte. Skriptid katavad andmestiku ettevalmistamise, vektoresituste genereerimise, PostgreSQL-i ja pgvectori jõudlustestid ning ligikaudse vektoriotsingu kvaliteedi hindamise recall@20 mõõdikuga.

## Kasulikud lingid

- [HuffPost News Category Dataset Kaggle'is](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- [pgvectori installatsioonijuhend](https://github.com/pgvector/pgvector/blob/master/README.md#installation)
- [pgvectori dokumentatsioon GitHubis](https://github.com/pgvector/pgvector)
- [PostgreSQL dokumentatsioon](https://www.postgresql.org/docs/)
- [Sentence Transformers dokumentatsioon](https://www.sbert.net/)
- [all-MiniLM-L6-v2 mudel Hugging Face'is](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [psycopg dokumentatsioon](https://www.psycopg.org/psycopg3/docs/)

## Failid

| Fail                    | Kirjeldus                                                                                                                                       |
|-------------------------|---|
| `schema_and_data.sql`   | Lisab pgvectori laienduse, loob PostgreSQL-i tabelid ja laadib CSV-failidest andmed tabelitesse. |
| `prepare_embeddings.py` | Puhastab HuffPosti uudisteandmestiku, loob 10 000, 25 000 ja 50 000 kirjega alamhulgad ning genereerib Sentence Transformersi mudeliga vektoresitused. |
| `pgvector_benchmark.py` | Käivitab PostgreSQL-i ja pgvectori jõudlustestid relatsiooniliste päringute, vektoriotsingu ja hübriidpäringute võrdlemiseks.                   |
| `pgvector_recall.py`    | Arvutab recall@20 mõõdiku HNSW, kohandatud HNSW ja IVFFlat indeksite jaoks, võrreldes ligikaudse otsingu tulemusi täpse vektoriotsinguga.       |
| `config.json`           | Sisaldab PostgreSQL-i ühenduse seadeid.                                                  |

## Nõuded

PostgreSQL 17 või uuem.

Python 3.12 või sellega ühilduv uuem Python 3 versioon.

Python-paketid saab paigaldada käsuga:

```bash
pip install pandas numpy sentence-transformers psycopg[binary]
```

Lisaks peab PostgreSQL-is olema kättesaadav `pgvector` laiendus. Installatsioonijuhend on leitav [pgvectori GitHubi lehel](https://github.com/pgvector/pgvector/blob/master/README.md#installation).

## Andmebaasi seadistus

Andmebaasi ühenduse seaded asuvad failis `config.json`.

Näide:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "dbname": "thesis_db",
    "user": "postgres",
    "password": ""
  }
}
```

Enne skriptide käivitamist tuleb failis `config.json` sisestada oma PostgreSQL-i andmebaasiühenduse aktuaalsed andmed.

`config.json` peab asuma samas kaustas, kus asuvad `pgvector_benchmark.py` ja `pgvector_recall.py`.

## 1. Andmestiku ettevalmistamine

Fail `prepare_embeddings.py` loeb [HuffPost News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) andmestiku, puhastab vajalikud väljad, ühendab pealkirja ja lühikirjelduse väljaks `content`, loob määratud suurusega alamhulgad ning genereerib vektoresitused.

Käivitamise näide:

```bash
python prepare_embeddings.py --input "C:/path/News_Category_Dataset_v3.json" --output-dir "C:/path/prepared_data"
```

Windowsi näide:

```bash
python prepare_embeddings.py --input "C:\path\News_Category_Dataset_v3.json" --output-dir "C:\path\prepared_data"
```

Olulisemad parameetrid:

| Parameeter | Selgitus |
|---|---|
| `--input` | Algandmestiku faili asukoht. |
| `--output-dir` | Kaust, kuhu väljundfailid salvestatakse. |
| `--model-name` | Kasutatav Sentence Transformersi mudel. Vaikimisi `sentence-transformers/all-MiniLM-L6-v2`. |
| `--batch-size` | Mitme teksti kaupa embeddinguid genereeritakse. |
| `--device` | Seade, näiteks `cpu` või `cuda`. Kui väärtust ei anta, valitakse see automaatselt. |
| `--seed` | Juhuarvu seeme alamhulkade reprodutseeritavaks loomiseks. |
| `--subset-sizes` | Loodavate alamhulkade suurused. Vaikimisi `10000 25000 50000`. |
| `--save-full-cleaned` | Salvestab lisaks kogu puhastatud andmestiku. |
| `--verbose` | Kuvab detailsema logi. |

Vaikimisi luuakse väljundkaustad:

```text
prepared_data/
  subset_10000/
    news_clean.csv
    news_with_embeddings_pgvector.csv
    metadata.json
  subset_25000/
    news_clean.csv
    news_with_embeddings_pgvector.csv
    metadata.json
  subset_50000/
    news_clean.csv
    news_with_embeddings_pgvector.csv
    metadata.json
```

Fail `news_with_embeddings_pgvector.csv` on mõeldud PostgreSQL-i laadimiseks tabelisse, mille vektoresituse veerg on näiteks tüüpi `VECTOR(384)`.

## 2. Andmebaasi tabelite loomine ja andmete laadimine

Fail `schema_and_data.sql` loob pgvectori laienduse, vajalikud tabelid ning sisaldab `COPY` käske CSV-failide laadimiseks PostgreSQL-i.

Enne faili käivitamist tuleb `COPY` käskudes muuta CSV-failide asukohad vastavalt oma arvuti failisüsteemile. Failid tekivad pärast `prepare_embeddings.py` käivitamist kaustadesse `prepared_data/subset_10000`, `prepared_data/subset_25000` ja `prepared_data/subset_50000`.

## 3. Jõudlustestid

Fail `pgvector_benchmark.py` käivitab kontrollitud jõudlustestid PostgreSQL-i ja pgvectori keskkonnas.

Põhivõrdlus sisaldab:

1. relatsioonilisi filtreid ilma B-puu indeksita;
2. relatsioonilisi filtreid B-puu indeksiga;
3. täpset vektoriotsingut ilma vektoriindeksita;
4. HNSW vektoriotsingut;
5. kohandatud HNSW vektoriotsingut;
6. IVFFlat vektoriotsingut;
7. relatsioonilise filtriga algavat hübriidotsingut.

Vektoriotsinguga algav hübriidotsing on eraldi katse, sest pärast metaandmete filtreerimist võib tagastatud ridade arv jääda väiksemaks kui etteantud `LIMIT`.

Käivitamine:

```bash
python pgvector_benchmark.py
```

Vaikimisi eeldab skript järgmisi tabeleid:

```text
public.news_articles_10k
public.news_articles_25k
public.news_articles_50k
```

Olulisemad kasutatavad veerud:

```text
id
category
published_date
embedding
```

Vaikimisi väljundkaust:

```text
pgvector_benchmark_results/
```

Loodavad failid:

| Fail | Kirjeldus |
|---|---|
| `main_raw_results.jsonl` | Põhivõrdluse detailsed tulemused. |
| `main_summary.csv` | Põhivõrdluse koondtulemused analüüsimiseks. |
| `main_plan_warnings.csv` | Põhivõrdluse juhtumid, mille täitmisplaani või mõõtmistingimusi tuleb kontrollida. |
| `vector_first_raw_results.jsonl` | Vektoriotsinguga algava hübriidotsingu detailsed tulemused. |
| `vector_first_summary.csv` | Vektoriotsinguga algava hübriidotsingu koondtulemused. |
| `vector_first_plan_warnings.csv` | Vektoriotsinguga algava hübriidotsingu hoiatused. |

Hoiatused näitavad, et tulemust tuleb tõlgendada ettevaatlikult. Näiteks võib hoiatus tekkida siis, kui PostgreSQL ei kasutanud oodatud indeksit, kui tulemusi tagastati vähem kui `LIMIT` või kui osa andmeplokke loeti kettalt.

## 4. Recall@20 hindamine

Fail `pgvector_recall.py` hindab ligikaudse vektoriotsingu kvaliteeti recall@20 mõõdiku abil.

Recall@20 arvutatakse täpse otsingu top-20 tulemuste ja ligikaudse otsingu top-20 tulemuste põhjal:

```text
recall@20 = | exact_top20_ids ∩ approximate_top20_ids | / 20
```

Skript võrdleb kolme ligikaudse vektoriotsingu konfiguratsiooni:

1. HNSW;
2. kohandatud HNSW;
3. IVFFlat.

Käivitamine:

```bash
python pgvector_recall.py
```

Vaikimisi väljundkaust:

```text
pgvector_recall_results/
```

Loodavad failid:

| Fail | Kirjeldus |
|---|---|
| `recall_raw_results.jsonl` | Detailsed tulemused iga päringuvektori ja indeksikonfiguratsiooni kohta. |
| `recall_summary.csv` | Koondtulemused recall@20 väärtuste, kattuvuse ja täitmisaegade kohta. |
| `recall_plan_warnings.csv` | Read, mille puhul tekkisid täitmisplaani hoiatused. |

Recall@20 on tehniline kattuvusmõõdik. See näitab, kui palju ligikaudse otsingu tulemused kattuvad täpse otsingu tulemustega, kuid ei mõõda otseselt tulemuste sisulist kasulikkust.

## Töövoog

1. Laadi alla HuffPost News Category Dataset.
2. Käivita `prepare_embeddings.py`, et luua puhastatud CSV-failid ja embeddingud.
3. Muuda failis `schema_and_data.sql` CSV-failide asukohad vastavalt oma keskkonnale ning käivita SQL-fail tabelite loomiseks ja andmete laadimiseks.
4. Kontrolli `config.json` andmebaasiühenduse seadeid.
5. Käivita `pgvector_benchmark.py`, et saada jõudlustestide tulemused.
6. Käivita `pgvector_recall.py`, et hinnata ligikaudse vektoriotsingu recall@20 väärtusi.
7. Kasuta `*_summary.csv` faile tulemuste analüüsimiseks ning `*_plan_warnings.csv` faile mõõtmiste kontrollimiseks.
