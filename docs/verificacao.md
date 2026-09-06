# Verificação executada em 6 de setembro de 2026

- Python 3.12, Windows, sem CUDA disponível.
- 14 testes automatizados aprovados: curadoria, deduplicação, separação por documento, PubMedQA sem vazamento da conclusão, métricas, máscara de perda, bloqueio, aprovação, rejeição, revisor obrigatório, isolamento, erros, alertas e interface Streamlit.
- Instalação das dependências verificada com `pip check`, sem conflitos.
- MedQuAD real, commit `577bd37b96c02d1833b2c9eed2de9f96964e96cb`: amostra de 3.000 exemplos, sendo 2.463 treino, 260 validação e 277 teste. Excluídos 5.777 arquivos das coleções 10/11/12; removidos 260 pares por tamanho e 1.991 perguntas duplicadas antes da amostragem. As contagens de arquivos e pares são unidades diferentes.
- PubMedQA real: 1.000 registros PQA-L convertidos para avaliação complementar.
- Teste de execução LoRA em CPU: arquitetura Qwen2 minúscula com pesos aleatórios e tokenizer Qwen2.5, um passo de treinamento, validação, salvamento, recarga do adapter e geração concluídos.
- O avaliador também completou a comparação base/adapter desse modelo aleatório, com dois casos por dataset. Esses resultados só verificam a execução e não representam desempenho médico; não foram incluídos como métricas do projeto.

Ainda não executado: QLoRA em GPU com o modelo Qwen2.5-1.5B-Instruct completo, avaliação de qualidade após o fine-tuning e revisão clínica humana. Esses passos devem ser realizados no notebook Colab antes da entrega acadêmica. O vídeo também precisa ser gravado.

## Correcao e teste no Colab

O Colab com Python 3.13 apresentava incompatibilidade entre torch 2.8.0 e torchvision 0.26.0 preinstalado, impedindo a importacao de Trainer. A instalacao passou a usar um venv sem pacotes globais. pip check e as importacoes passaram nesse ambiente. Foram executados um passo real de QLoRA com Qwen2.5-1.5B-Instruct na Tesla T4, validacao em dois exemplos e salvamento do adapter de diagnostico em models/qlora-smoke. O treinamento completo e a avaliacao de qualidade continuam pendentes. Os testes automatizados passaram de 14 para 16, incluindo a propagacao de erros de subprocessos.


## Evidencias recebidas em 2026-09-06

As pendencias de treinamento e avaliacao mencionadas acima foram concluidas: o ZIP entregue contem o adapter final, manifestos e respostas de 50 casos por dataset/modelo. A integridade do ZIP e os 112 tensores foram verificados; as metricas foram recalculadas e coincidem com comparison.json. A analise atual esta no relatorio tecnico. Revisao humana e video permanecem pendentes.
