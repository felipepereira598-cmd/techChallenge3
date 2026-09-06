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
| Fine-tuning com dados médicos | scripts/train.py e notebook Colab; execução QLoRA concluída |
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
| Avaliação e análise | evaluate.py; comparação real registrada abaixo |
| Vídeo de até 15 minutos | roteiro_video.md; gravação pendente |

## Resultados observados em 6 de setembro de 2026

Evidências importadas de training-artifacts.zip. Integridade do ZIP e leitura dos 112 tensores do adapter verificadas. As quatro métricas foram recalculadas a partir dos JSONL e coincidem com comparison.json. Os dois modelos responderam aos mesmos IDs e usam a revisão 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.

Treinamento QLoRA na Tesla T4, uma época: 2.430 exemplos retidos de 2.471 entradas, 41 descartados por comprimento; validação com 259 de 260 entradas. Tempo de treino: 1.270,48 segundos. Loss de treino: 1,23985; loss de validação: 1,26210. O adaptador foi salvo e a avaliação completou 50 exemplos por dataset e por modelo.

| Métrica | Base | Fine-tuned |
|---|---:|---:|
| MedQuAD token F1 | 0,282087 | 0,295029 |
| MedQuAD respostas sinalizadas por regras | 10% (5/50) | 10% (5/50) |
| PubMedQA acurácia | 32% (16/50) | 20% (10/50) |
| PubMedQA respostas fora do formato | 0% | 0% |

O F1 lexical aumentou 0,01294, mas a acurácia complementar caiu 12 pontos percentuais, equivalente a seis acertos a menos. Portanto, o experimento demonstra execução do fine-tuning, sem evidência de ganho consistente. A amostra pequena é exploratória; F1 não mede correção clínica e as regras não detectam todos os riscos. A especialização em QA do MedQuAD pode contribuir para a diferença no PubMedQA, mas este experimento não estabelece a causa. A pouca cobertura hospitalar em português também limita a generalização.

As respostas individuais e manifestos estão preservados em results/ e models/adapter/ no pacote local. Ainda falta revisão humana pela rubrica, incluindo erros/regressões, e a gravação da demonstração. Não houve revisão clínica humana nesta verificação automatizada.

## Limitações e próximos passos exigidos para entrega

O projeto não foi validado para atendimento. Anonimização por regex é incompleta, revisão não tem autenticação e o log não é imutável. Revisões pendentes dependem da sessão. Modelos podem alucinar ou seguir instruções maliciosas apesar do prompt. A revisão humana é necessária e também pode errar. Antes da entrega acadêmica, concluir revisão humana dos resultados e gravar a demonstração. Nenhum resultado numérico de qualidade foi inventado neste relatório.
