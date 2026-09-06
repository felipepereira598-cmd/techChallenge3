# Relatório técnico — Tech Challenge Fase 3

## Objetivo e escopo

Assistente acadêmico para resumo de registros respiratórios sintéticos, verificação de exames pendentes e alerta para revisão. O sistema não executa condutas, receitas ou alterações em prontuários. A menor arquitetura adotada usa JSON, Streamlit, Transformers/PEFT, LangChain e LangGraph.

## Fine-tuning

MedQuAD é o corpus principal; exemplos sintéticos complementam os requisitos de protocolo, FAQ, modelos de laudo, receita e procedimento internos. Receitas são apenas templates vazios não executáveis. Dados públicos não são apresentados como dados reais do hospital.

A preparação normaliza texto, remove casos inadequados, redige padrões de identificadores e deduplica perguntas. O hash do documento define partição, evitando que perguntas do mesmo documento atravessem treino e teste. Isto não elimina paráfrases duplicadas entre documentos nem sobreposição com pré-treinamento. É necessária curadoria humana.

Treinamento padrão: Qwen2.5-1.5B-Instruct, QLoRA NF4 com dupla quantização, LoRA rank 8/alpha 16/dropout 0,05 nas projeções q/v. Batch 1 e acumulação 8, learning rate 2e-4, uma época. A máscara de perda ignora o prompt; exemplos acima de 1.024 tokens são descartados e registrados. Validação ocorre ao final da época, e o adapter é salvo sem copiar os pesos base.

O modelo pequeno reduz custo no Colab, com limitação de capacidade. A mistura MedQuAD em inglês e exemplos em português deve ser considerada na avaliação: melhorar QA em inglês não garante desempenho clínico em português. Oito exemplos internos apenas demonstram cobertura e não permitem concluir domínio dos processos hospitalares.

## Assistente e fluxo

O diagrama no README documenta o fluxo LangChain/LangGraph. Tools consultam paciente e protocolo por ID. Um pipeline Runnable conecta o prompt à LLM customizada. Um nó verifica padrões proibidos; saídas sinalizadas terminam bloqueadas. As demais param com interrupt até aprovação ou rejeição humana. O estado e os registros consultados são exibidos para conferência.

Logs detalham as etapas e o resultado da revisão. A fonte é atribuída pelo sistema a registros realmente consultados, não por URL inventada pela LLM. Esta explicabilidade é rastreabilidade de evidência e observações, não exposição de cadeia de pensamento ou garantia de causalidade.

## Matriz de requisitos do PDF

| Obrigação | Implementação / evidência |
|---|---|
| Fine-tuning com dados médicos | scripts/train.py e notebook Colab; execução real pendente |
| Protocolos, FAQs e documentos internos | data/synthetic/training.jsonl, categorias explícitas |
| Preprocessing, anonimização e curadoria | prepare_medquad.py, safety.anonymize, revisão de amostra |
| LLM customizada integrada ao LangChain | src/llm.py, LocalLLM e build_chain |
| Consultas estruturadas e contexto do paciente | src/tools.py e nó load |
| Fluxos LangGraph | src/graph.py, decisão condicional e interrupt |
| Limites de atuação e validação humana | safety.py, nó human_review e interface |
| Logging detalhado | logger.py, logs/audit.jsonl |
| Explainability com fonte | fontes do estado, protocolo e registros exibidos |
| Python modular e README | src, scripts, tests e README |
| Dataset sintético | data/synthetic |
| Relatório e diagrama | este documento e README |
| Avaliação e análise | evaluate.py; preencher análise após rodar Colab |
| Vídeo de até 15 minutos | roteiro_video.md; gravação pendente |

## Avaliação planejada e análise a completar

Execute os modelos base e adapter no mesmo teste MedQuAD e na mesma amostra PQA-L. Geração greedy e revisão fixada controlam parte da variabilidade. Os resultados individuais permitem inspecionar regressões, além das médias. Não escolher casos de teste após observar resultados.

| Evidência | Situação inicial |
|---|---|
| Treino GPU, loss e contagens de tokenização | A executar no Colab |
| MedQuAD token F1 base/fine-tuned | A gerar em results/comparison.json |
| PubMedQA acurácia e taxa inválida | A gerar em results/comparison.json |
| Revisão humana de fidelidade, completude e segurança | A preencher na rubrica |
| Conclusão sobre benefício do fine-tuning | Indeterminada antes do experimento |

Após execução: registrar GPU, tempo, versões, revisão do modelo, commits dos datasets, contagens por split e descartes. Comparar valores, revisar no mínimo dez erros/regressões e discutir idioma, representatividade e tamanho amostral. Manter resultados negativos. Uma amostra de 50 exemplos é exploratória; não demonstra eficácia ou segurança clínica. A taxa de padrões de risco mede o detector, não todas as violações possíveis.

## Limitações e próximos passos exigidos para entrega

O projeto não foi validado para atendimento. Anonimização por regex é incompleta, revisão não tem autenticação e o log não é imutável. Revisões pendentes dependem da sessão. Modelos podem alucinar ou seguir instruções maliciosas apesar do prompt. A revisão humana é necessária e também pode errar. Antes da entrega acadêmica, executar treino real, baixar evidências, preencher análise e gravar a demonstração. Nenhum resultado numérico de qualidade foi inventado neste relatório.
