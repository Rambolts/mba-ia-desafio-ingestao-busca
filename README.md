# Desafio MBA Engenharia de Software com IA - Full Cycle

Projeto de estudo de RAG que ingere um documento PDF, armazena seus embeddings no PostgreSQL com pgvector e responde a perguntas pelo terminal.

## Pré-requisitos

- Python 3.10 ou superior.
- Docker com o plugin Docker Compose.
- Uma chave de API da OpenAI.

## Configuração

Crie o ambiente virtual e instale as dependências:

```bash
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -r requirements.txt
```

Se o arquivo `.env` ainda não existir, crie-o a partir do exemplo:

```bash
if [ ! -f .env ]; then
    cp .env.example .env
fi
```

Configure no `.env` os valores abaixo. Informe sua chave de API sem compartilhá-la ou adicioná-la ao Git:

```dotenv
OPENAI_API_KEY=sua-chave-de-api
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=localhost
PG_VECTOR_COLLECTION_NAME=document_chunks
PDF_PATH=document.pdf
```

`PDF_PATH` pode ser absoluto ou relativo à raiz do projeto. Use o mesmo caminho ao ingerir e consultar o documento, pois ele identifica seus trechos no banco. O valor `localhost` usa as credenciais locais definidas em `docker-compose.yml`.

O fluxo atual usa a OpenAI para embeddings e respostas. As variáveis `GOOGLE_*` do arquivo de exemplo ainda não são usadas pelo chat nem pela ingestão.

## Iniciar o banco de dados

Na raiz do projeto, inicie os serviços:

```bash
docker compose up -d
```

O Compose inicia o PostgreSQL 17 com pgvector e cria a extensão `vector`. Na primeira ingestão, `src/ingest.py` também cria a tabela configurada em `PG_VECTOR_COLLECTION_NAME`, caso ainda não exista.

## Ingerir o PDF

Coloque o documento indicado por `PDF_PATH` no caminho configurado e execute:

```bash
python src/ingest.py
```

O script extrai o texto, divide-o em chunks de até 1.000 caracteres com overlap de 150, gera os embeddings e grava o conteúdo e os vetores na tabela. Executar a ingestão novamente substitui os chunks anteriores desse mesmo caminho.

PDFs sem texto extraível, como documentos digitalizados sem OCR, não podem ser ingeridos diretamente.

## Fazer perguntas

Com o banco em execução e o PDF ingerido, inicie o chat:

```bash
python src/chat.py
```

Digite uma pergunta no terminal. O sistema vetoriza a pergunta, busca até 10 chunks mais próximos do documento configurado e envia o contexto recuperado à LLM. Para encerrar, digite `sair` ou `exit`, ou use `Ctrl+C`.

Se não houver chunks para o documento, o chat informa que não tem dados suficientes para responder. A busca usa distância de cosseno e as regras do prompt instruem a LLM a responder somente com base no contexto.

## Depurar no VS Code

Abra a pasta do projeto no VS Code, selecione o interpretador Python do ambiente `.venv` e abra **Executar e Depurar**. Escolha uma das configurações existentes:

- `Python: Ingestão do PDF`
- `Python: Chat`

Pressione `F5` para iniciar. Ambas as configurações carregam o `.env` da raiz do projeto.
