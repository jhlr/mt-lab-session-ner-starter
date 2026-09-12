# mt-lab-session-ner-starter

## Grupo
1. Joao Rietra - jhlr@cesar.school

## Dataset escolhido
`cellphone.ibyte.json` (1326 produtos, 1034 títulos únicos). Categoria de celulares/smartphones e acessórios (capas, carregadores, suportes, películas).

## Schema de entidades (NER)
| Tag | Descrição | Exemplo |
|---|---|---|
| `TIPO` | Tipo/categoria do produto | Smartphone, Celular, Capa, Carregador, Case, Suporte, Película |
| `MARCA` | Marca | Apple, Samsung, Motorola, Nokia, Multilaser |
| `MODELO` | Linha/modelo do produto | iPhone 14 Pro Max, Galaxy A53, Moto E22 |
| `MEMORIA` | Capacidade de armazenamento | 128GB, 64GB |
| `RAM` | Memória RAM | 4GB RAM, 8GB de RAM |
| `COR` | Cor | Preto, Roxo-profundo, Rose Gold |
| `TELA` | Tamanho de tela | Tela 6.5, Tela de 6,5 |

Códigos de SKU/referência (ex: `P9076`, `MM2Y3ZE/A`) foram deixados sem tag.

## Metodologia de anotação
Amostra aleatória de 300 títulos únicos (seed=42), anotada com apoio de LLM em 5 lotes de 60 títulos. Cada offset foi validado programaticamente contra o texto original (sem overlaps, sem offsets inválidos). Script de amostragem em `scripts/2.sample_titles.py`, pra reproduzir a mesma amostra. Anotações finais em `data/annotations/cellphone.ibyte.jsonl`.

## Prerequisites:
* Docker
* Python

## Overview
1. Considere o cenário de uma plataforma de e-commerce, no fluxo de cadastrar novos produtos.
2. Seu objetivo é construir um sistema para auxiliar a extração de informações de um produto, a partir do título dele. Assim, as informações de catálogo/cadastro do produto podem ser preenchidas automaticamente.
3. Por exemplo, quando o usuário digitar "iphone 14 128gb vermelho", o sistema já deve identificar e sugerir automaticamente a categoria (SMARTPHONES), o modelo (IPHONE 14), a memória (128GB) e a cor (VERMELHO).

## Descrição da atividade
1. Cada grupo deve escolher uma base de dados na pasta `data/raw/` e escolher uma base de dados (diferente de `beauty.ibyte.jsonl` e `toys_and_babies.ibyte`)
2. O grupo deve realizar o processo de definição das TAGs (entidades nomeadas), anotação, treinamento e seleção da melhor abordagem de NER para resolver esse problema (ver seção [Running Doccano](#running-doccano)). O grupo pode usar outras alternativas para anotação das entidades (datasets externos, zero-shot learning, etc).
4. As anotações devem ser salvas na pasta `data/annotations/nome_do_dataset.jsonl`, no mesmo formato dos arquivos de exemplo `beauty.ibyte.jsonl` e `toys_and_babies.ibyte`, e.g.:
```
{"text": "Pelucia Bonnie Bear 30Cm Azul Multikids   BR166", "entities": [[0, 7, "TIPO"], [8, 19, "PRODUTO"], [30, 39, "MARCA"]]}
```
5. Os notebooks para treinamento e avaliação devem estar salvos de forma ORGANIZADA na pasta `notebooks`
6. No dia agendado para a apresentação, o grupo deve mostrar em sala como foi o processo de solução do problema, e os resultados obtidos (no máximo 5 - 10 min por grupo).

## Avaliação:
Os critérios para a avaliação serão:
- Anotações das entidades no formato adequado
- Organização e metodologia experimental
- Apresentação dos resultados

# Running Doccano
## Run as docker container
```
chmod +x scripts/start-doccano.sh
./scripts/0.setup-doccano.sh
./scripts/1.start-doccano.sh
```

## Import annotations
1. Access doccano at the URL localhost:8000
2. Login 
3. Go to "Dataset" --> "Actions" --> Import Dataset
4. Change the File Format to JSONL
5. Set column label to be "entities"

![Doccano Setup](imgs/doccano-import-dataset.png)

6. See the annotations at "Start Annotation"