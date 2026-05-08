# Relatório — Coverage Path Planning com PPO

Objetivo do projeto: um agente num **grid com obstáculos** tem de **passar por todas as células livres** (coverage path planning). Ele **não vê o mapa inteiro** — só uma **janelinha centrada nele** (regra da APS: 3×3 ou 5×5).

---

## 1. O problema em poucas palavras

- **Ações:** cima, baixo, esquerda, direita (4).
- **Recompensa (resumo):** ganha ao visitar célula nova; perde ao repetir célula, ao bater na parede ou ao gastar um passo; bónus grande se cobrir **tudo**; penalidade se o episódio acabar por limite de passos sem ter coberto tudo.


### Baseline do enunciado (janela 3×3, treino simples)

Treino em 5×5; depois teste em 10×10 sem treinar de novo nesse tamanho (“zero-shot”).

| Grid | Episódios com cobertura **total** (5 testes × 100 episódios) | Média |
|------|-------------------------------------------------------------|-------|
| 5×5 | resultados entre runs ~69–81% dos episódios | ~76% |
| 10×10 | pior | ~64% |

Conclusão rápida: **não generaliza bem** para o mapa maior; mesmo no 5×5 falha muitas vezes.

---

## 2. Porque é difícil

- **Janela pequena:** no 10×10 o agente vê só um recorte do mundo; longe das últimas células por visitar, **não há informação** para saber para onde ir.
- **Rede “normal” (MLP):** tratar a matriz da janela como um monte de números em fila **não ajuda** a rede a entender que é um **quadrado 2D** (paredes, cantos, etc.).
- **Treinar logo no 10×10 do zero** é mais difícil: o agente demora a receber recompensas boas.

---

## 3. Experiência A — Janela maior (5×5 em vez de 3×3)

**Ideia:** sem mudar a regra da APS, usar a **maior janela permitida (5×5)** para o agente ver mais à volta.

**O que aconteceu (média de 5 testes, 100 episódios cada):**

| Modelo | Treino | 5×5 cobertura total | 10×10 cobertura total |
|--------|--------|---------------------|------------------------|
| A | Só 5×5, do zero | **~94%** | — |
| B | Só 10×10, do zero | — | **~61%** |

No 5×5 melhorou **muito** face ao baseline (~76%). No 10×10 ainda **falha** em muitos episódios: muitas vezes visita **quase** todas as células mas **não fecha** o último pedaço antes do limite de passos.


---

## 4. Experiência B — Curriculum (treino em etapas)

**Ideia:** em vez de começar pelo mapa difícil, **treinar primeiro num caso mais fácil** e **continuar o treino** com o mesmo modelo noutro cenário.

- **Modelo C:** primeiro 5×5 **sem obstáculos**, depois 5×5 com obstáculos.
- **Modelo D:** treinar no 5×5 (como o modelo A), depois **continuar** no 10×10 com os mesmos pesos iniciais.

**Resultados (média de 5 testes):**

| Modelo | 5×5 | 10×10 |
|--------|-----|-------|
| C | **~96%** | — |
| D | — | **~80%** |

Ou seja: **treinar primeiro no 5×5 e só depois no 10×10** ajudou bastante no 10×10 em relação ao modelo B (~61%).

**Resumo numa tabela:**

| Abordagem | 5×5 (média) | 10×10 (média) |
|-----------|-------------|----------------|
| Baseline enunciado | ~76% | ~64% (zero-shot) |
| Janela 5×5, do zero | ~94% | ~61% |
| Curriculum (modelo D) | — | **~80%** |


---

## 5. Experiência C — Rede com “visão” + memória (RecurrentPPO + CNN + curriculum)

**Ideia em linguagem simples:**

1. **CNN** — olha para a janela como uma **imagem pequena** (padrões locais: parede, sítios por visitar).
2. **LSTM** — guarda **informação ao longo do episódio** (útil quando a última célula está **fora** da janela atual).
3. **Treino em etapas:** 5×5 (1,5M passos) → 10×10 (2M passos), **carregando** o modelo anterior.

**Comandos:**

```bash
source venv/bin/activate

python train_grid_world_cpp.py train --stage 5
python train_grid_world_cpp.py train --stage 10 --from data/recppo_stage5_<data_e_hora>
python train_grid_world_cpp.py train --stage 20 --from data/recppo_stage10_<data_e_hora>   # opcional

python train_grid_world_cpp.py eval_all --model data/recppo_stage10_<data_e_hora>
```

**Resultados (`eval_all`, 100 episódios por grid, sem `--deterministic`):**

| Grid | Cobertura total (“full”) | Cobertura média | Passos médios |
|------|--------------------------|-----------------|----------------|
| 5×5 (3 obst., cap 200) | **95 / 100** | 99,36% | ~34 |
| 10×10 (12 obst., cap 400) | **77 / 100** | 99,00% | ~178 |


**Leitura rápida:** no 10×10 a **cobertura média** (~99%) está quase sempre “perto do fim”, mas só em **~77%** dos episódios chega a **100%** antes do limite de passos — o mesmo tipo de trade-off que já víamos na experiência B (quase tudo visitado vs. fecho completo). Fica **próximo** do ~80% do modelo D (PPO + curriculum), com uma receita diferente (CNN + LSTM).

---

## 6. Se ainda não chegar (opcional)

Ajustar **recompensas** (por exemplo tornar menos penalizante ficar sem cobrir tudo até ao fim do episódio, ou dar um pequeno incentivo a **explorar sítios novos** visíveis na janela). Só faz sentido depois de ver os números da experiência C.

---

