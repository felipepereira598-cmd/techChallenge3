# Assistente hospitalar acadÃªmico â€” Tech Challenge Fase 3

Projeto Python mÃ­nimo para o desafio 8IADT: MedQuAD + exemplos hospitalares sintÃ©ticos â†’ LoRA/QLoRA no Colab â†’ assistente LangChain/LangGraph â†’ revisÃ£o humana â†’ auditoria JSONL.

**Somente demonstraÃ§Ã£o acadÃªmica. NÃ£o usar para assistÃªncia real.** Os pacientes, protocolos e documentos internos sÃ£o fictÃ­cios. As regras nÃ£o constituem protocolos clÃ­nicos validados. O modo demonstraÃ§Ã£o nÃ£o executa LLM e nÃ£o comprova fine-tuning.

## ExecuÃ§Ã£o rÃ¡pida (sem GPU)

Python 3.11 ou 3.12. Na raiz do repositÃ³rio:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

Selecione P001 e analise as pendÃªncias. A resposta fica em `pending_review` atÃ© registrar uma decisÃ£o com identificador de revisor. P003 demonstra um alerta fictÃ­cio. Rejeitar encerra sem liberar a resposta. SaÃ­das com padrÃµes proibidos ficam bloqueadas sem botÃ£o de aprovaÃ§Ã£o. Nenhuma ferramenta emite receita ou modifica prontuÃ¡rio.

## Fine-tuning real no Colab

Abra [o notebook no Google Colab](https://colab.research.google.com/github/felipepereira598-cmd/techChallenge3/blob/main/notebooks/training_colab.ipynb), ative GPU T4 e execute em ordem. O notebook clona este repositÃ³rio, instala dependÃªncias, baixa as fontes e prepara cerca de 3.000 exemplos. RepositÃ³rio privado requer token de leitura informado em campo oculto; nÃ£o o salve em cÃ³digo.

1. MedQuAD: XML â†’ limpeza de espaÃ§os, remoÃ§Ã£o de respostas ausentes/longas, exclusÃ£o das coleÃ§Ãµes 10/11/12 e deduplicaÃ§Ã£o das perguntas.
2. SeparaÃ§Ã£o determinÃ­stica por documento-fonte, aproximadamente 80/10/10; distribuiÃ§Ã£o exata fica no manifesto.
3. Curadoria humana da amostra, preservaÃ§Ã£o de URL e tipo da pergunta; anonimizaÃ§Ã£o bÃ¡sica por padrÃµes. **Nomes e outros identificadores em texto livre exigem revisÃ£o manual. NÃ£o alimentar dados reais.**
4. Treino combina MedQuAD com `data/synthetic/training.jsonl`, incluindo protocolo, FAQ, laudo, receita vazia e registro de procedimento. Esses exemplos sÃ£o mÃ­nimos para demonstrar cobertura, nÃ£o um corpus hospitalar representativo.
5. Qwen2.5-1.5B-Instruct com QLoRA NF4, rank 8, q/v projections, 1 Ã©poca, batch 1, acumulaÃ§Ã£o 8, sequÃªncia 1.024. A perda Ã© calculada somente na resposta do assistente. Exemplos maiores sÃ£o descartados e contados, sem cortar respostas silenciosamente.
6. Salvar adapter e manifestos. A revisÃ£o do modelo Ã© resolvida e fixada no inÃ­cio do notebook. Os commits dos dados e hashes dos arquivos de entrada ficam registrados.
7. Comparar modelo base e fine-tuned nas mesmas perguntas de teste, com a mesma revisÃ£o e geraÃ§Ã£o determinÃ­stica. PubMedQA PQA-L Ã© usado sÃ³ para avaliaÃ§Ã£o complementar, nunca no treino.
8. Baixar `training-artifacts.zip`, extrair `models/adapter` na raiz local e reiniciar o aplicativo.

O tempo e a memÃ³ria dependem da GPU e do comprimento dos exemplos. Para diagnÃ³stico rÃ¡pido, acrescente `--max-steps 2` ao comando de treino; isso Ã© teste de execuÃ§Ã£o, nÃ£o experimento final. Se faltar memÃ³ria, reduza `--max-length` para 512 e registre o aumento dos descartes. Uma reinicializaÃ§Ã£o do Colab pode ser necessÃ¡ria apÃ³s a instalaÃ§Ã£o.

## Usar o adapter na interface

```sh
python -m pip install -r requirements-training.txt
python -m streamlit run app.py
```

Selecione **Fine-tuned local**. O adapter padrÃ£o Ã© `models/adapter`; a variÃ¡vel `ADAPTER_PATH` permite outro diretÃ³rio. InferÃªncia local usa CPU quando nÃ£o hÃ¡ CUDA, com possÃ­vel lentidÃ£o. O aplicativo falha explicitamente se o adapter estiver ausente. NÃ£o substitui um adapter faltante pelo modelo base. O modo local lÃª automaticamente o modelo e a revisÃ£o do manifesto salvo com o adapter. A avaliaÃ§Ã£o usa esse mesmo manifesto para manter a comparaÃ§Ã£o consistente.

## Comandos independentes

```sh
python -m scripts.prepare_medquad --raw data/raw/MedQuAD --limit 3000
python -m scripts.prepare_pubmedqa --input data/raw/pubmedqa/data/ori_pqal.json
python -m scripts.train --qlora --revision COMMIT_DO_MODELO
python -m scripts.evaluate --quantized --revision COMMIT_DO_MODELO --limit 50
```

`--qlora` requer GPU CUDA, Linux e bitsandbytes. Sem essa opÃ§Ã£o o treino usa LoRA, sem quantizaÃ§Ã£o. O notebook usa subprocessos com verificaÃ§Ã£o de erro para treino e avaliaÃ§Ã£o.

## Fluxo e fontes

```mermaid
flowchart TD
    A[Streamlit: pergunta e ID sintÃ©tico] --> B[LangGraph: carregar registros JSON]
    B --> C[Tools LangChain: paciente e protocolo por ID]
    C --> D[Pipeline LangChain: prompt e LLM com adapter]
    D --> E[Regras de seguranÃ§a]
    E -->|violaÃ§Ã£o| F[Bloqueado]
    E -->|sem padrÃ£o proibido| G[Interrupt: revisÃ£o humana]
    G -->|aprovar| H[Resposta revisada]
    G -->|rejeitar| I[Resposta rejeitada]
    B -.-> J[Log JSONL]
    D -.-> J
    E -.-> J
    G -.-> J
```

Consultas estruturadas usam IDs exatos em JSON; nÃ£o hÃ¡ RAG, embeddings, banco vetorial, API de serviÃ§o ou orquestraÃ§Ã£o distribuÃ­da. O estado Ã© atualizado a cada nova solicitaÃ§Ã£o. A revisÃ£o considera o snapshot apresentado; se os registros mudarem, inicie nova anÃ¡lise.

A interface exibe observaÃ§Ãµes, exames pendentes, alertas e fontes com caminho, ID e versÃ£o/data. Isso oferece rastreabilidade do contexto fornecido, **nÃ£o uma prova automÃ¡tica de suporte a cada frase nem acesso ao raciocÃ­nio interno da LLM**. URLs lembradas pelo modelo nÃ£o sÃ£o tratadas como evidÃªncia validada. O revisor deve verificar fidelidade Ã s fontes apresentadas.

## Auditoria e seguranÃ§a

`logs/audit.jsonl` registra timestamp UTC, ID da solicitaÃ§Ã£o, etapas, dados sintÃ©ticos consultados, pergunta redigida, fontes, rascunho, modelo, versÃ£o do prompt, duraÃ§Ã£o, violaÃ§Ãµes, decisÃ£o e observaÃ§Ã£o do revisor. Falhas registram tipo de erro, sem despejar exceÃ§Ãµes que possam conter dados. O log Ã© local, nÃ£o tem criptografia ou proteÃ§Ã£o contra alteraÃ§Ã£o e nÃ£o deve receber dados reais. NÃ£o o publique automaticamente.

O checkpointer em memÃ³ria mantÃ©m a pausa durante a sessÃ£o. Reiniciar o servidor perde revisÃµes pendentes; o log permanece. A identificaÃ§Ã£o do revisor Ã© autodeclarada, adequada somente Ã  demonstraÃ§Ã£o local. NÃ£o hÃ¡ autenticaÃ§Ã£o profissional. Regex tem falsos positivos e negativos; nenhuma aprovaÃ§Ã£o Ã© automÃ¡tica. ConteÃºdo do paciente Ã© tratado como dado no prompt, mas isso nÃ£o elimina prompt injection. Toda resposta liberada exige revisÃ£o humana.

## AvaliaÃ§Ã£o e entregÃ¡veis

`results/base.jsonl` e `results/fine_tuned.jsonl` preservam respostas individuais. `results/comparison.json` contÃ©m token F1 MedQuAD, taxa de padrÃµes de risco, acurÃ¡cia PubMedQA e respostas fora do formato. Token F1 mede sobreposiÃ§Ã£o lexical, nÃ£o correÃ§Ã£o mÃ©dica. PubMedQA mede resposta ao resumo cientÃ­fico, nÃ£o conduta clÃ­nica. O conjunto PQA-L completo/amostra Ã© uma avaliaÃ§Ã£o local, nÃ£o o protocolo oficial de benchmark. FaÃ§a revisÃ£o cega adicional usando `data/evaluation/human_rubric.json`.

**NÃ£o hÃ¡ resultado de treinamento prÃ©-calculado nem ganho de qualidade declarado.** Execute o notebook e incorpore os resultados reais ao [relatÃ³rio tÃ©cnico](docs/relatorio_tecnico.md). O [roteiro de vÃ­deo](docs/roteiro_video.md) cobre a apresentaÃ§Ã£o de atÃ© 15 minutos. O cÃ³digo implementa o pipeline; a entrega acadÃªmica sÃ³ estarÃ¡ completa apÃ³s treinamento, anÃ¡lise de resultados e gravaÃ§Ã£o.

## Estrutura

- `src/`: LLM, tools, graph, safety e logger.
- `scripts/`: preparaÃ§Ã£o MedQuAD/PubMedQA, treino e avaliaÃ§Ã£o.
- `notebooks/`: execuÃ§Ã£o Colab.
- `data/synthetic/`: registros hospitalares e exemplos de treino fictÃ­cios.
- `data/evaluation/`: rubrica humana independente.
- `tests/`: revisÃ£o, bloqueio, isolamento, logging, curadoria e mÃ¡scara de perda.
- `docs/`: relatÃ³rio, matriz de requisitos e roteiro de demonstraÃ§Ã£o.

## Fontes e licenÃ§as

- [MedQuAD](https://github.com/abachaa/MedQuAD): Ben Abacha e Demner-Fushman (2019), *A Question-Entailment Approach to Question Answering*. Dataset CC BY 4.0 conforme repositÃ³rio. ColeÃ§Ãµes com respostas removidas nÃ£o sÃ£o recuperadas por crawler. Preservar atribuiÃ§Ã£o ao distribuir derivados.
- [PubMedQA](https://github.com/pubmedqa/pubmedqa): Jin et al. (2019), *PubMedQA: A Dataset for Biomedical Research Question Answering*. Consultar termos do dataset e fontes antes de redistribuir os resumos; baixados no Colab e nÃ£o incorporados aqui.
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): consultar model card e licenÃ§a do modelo antes de distribuir adapters.
- [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization) e [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts).

Os dados processados sÃ£o reconstruÃ­dos pelo pipeline e ficam fora do Git por padrÃ£o. Os exemplos sintÃ©ticos autorais ficam versionados. Isso satisfaz o entregÃ¡vel de exemplo de dataset sem duplicar datasets de terceiros.
