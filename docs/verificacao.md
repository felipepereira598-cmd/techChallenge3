# Verificação executada em 6 de setembro de 2026

- Python 3.12, Windows, sem CUDA disponível.
- 14 testes automatizados aprovados: curadoria, deduplicação, separação por documento, PubMedQA sem vazamento da conclusão, métricas, máscara de perda, bloqueio, aprovação, rejeição, revisor obrigatório, isolamento, erros, alertas e interface Streamlit.
- Instalação das dependências verificada com `pip check`, sem conflitos.
- MedQuAD real, commit `577bd37b96c02d1833b2c9eed2de9f96964e96cb`: amostra de 3.000 exemplos, sendo 2.463 treino, 260 validação e 277 teste. Excluídos 5.777 arquivos das coleções 10/11/12; removidos 260 pares por tamanho e 1.991 perguntas duplicadas antes da amostragem. As contagens de arquivos e pares são unidades diferentes.
- PubMedQA real: 1.000 registros PQA-L convertidos para avaliação complementar.
- Teste de execução LoRA em CPU: arquitetura Qwen2 minúscula com pesos aleatórios e tokenizer Qwen2.5, um passo de treinamento, validação, salvamento, recarga do adapter e geração concluídos.
- O avaliador também completou a comparação base/adapter desse modelo aleatório, com dois casos por dataset. Esses resultados só verificam a execução e não representam desempenho médico; não foram incluídos como métricas do projeto.

Ainda não executado: QLoRA em GPU com o modelo Qwen2.5-1.5B-Instruct completo, avaliação de qualidade após o fine-tuning e revisão clínica humana. Esses passos devem ser realizados no notebook Colab antes da entrega acadêmica. O vídeo também precisa ser gravado.
