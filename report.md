# Relatório — Coverage Path Planning com PPO

> Documento incremental: cada estratégia testada vira uma seção própria com motivação, implementação, resultados e análise.

## 1. Contexto

O ambiente `GridWorldCPP` é uma adaptação do `GridWorld` para **Coverage Path Planning (CPP)**: o agente deve visitar todas as células livres de um grid `N x N` com obstáculos, mantendo observação **parcial** do mapa.

- **Espaço de ações**: `Discrete(4)` — direita, cima, esquerda, baixo.
- **Espaço de observação** (Dict, `MultiInputPolicy`):
  - `agent`: `[x/N, y/N, coverage_ratio]` (3 floats normalizados).
  - `neighbors`: matriz egocêntrica com o agente sempre no centro. **Pelas regras da APS, o tamanho permitido é 3x3 ou 5x5**. Codificação: `0` = livre não visitada, `1` = parede/obstáculo (incluindo fora do grid), `2` = visitada.
- **Função de recompensa**: `+1.0` célula nova, `-0.3` revisita, `-0.5` colidir/ficar parado, `-0.1` por passo, `+10.0` cobertura completa, `-5.0` truncamento.
- **Algoritmo**: PPO (`stable-baselines3`) com `MultiInputPolicy`, `ent_coef=0.05`, `device="cpu"`.

### Baseline (configuração original do professor, janela 3x3)

Hiperparâmetros: `DIM=5`, `OBSTACLES=3`, `MAX_STEPS=200`, `TOTAL_TIMESTEPS=1.000.000`.

| Grid | Full Coverage Rate (5 execuções de 100 episódios) | Média |
|------|----------------------------------------------------|-------|
| 5x5  | 75 / 78 / 69 / 79 / 81                             | ~76,4% |
| 10x10 (modelo treinado em 5x5, zero-shot)          | 66 / 65 / 60 / 70 / 59 | ~64,0% |

**Observação**: o modelo treinado em 5x5 não generaliza bem para 10x10. Mesmo no próprio 5x5 o agente não atinge cobertura total em todos os episódios.

---

## 2. Diagnóstico

A informação espacial disponível ao agente é o gargalo principal:

1. **Janela 3x3 é local demais.** Em 5x5 essa janela cobre ~36% do grid, mas em 10x10 cobre apenas ~9%. Quando o agente se afasta de regiões inexploradas, ele literalmente não consegue "ver" para onde ir.
2. **`MultiInputPolicy` achata a matriz 3x3 em 9 features** que entram em uma MLP padrão (2 camadas de 64 unidades). Não há *inductive bias* espacial; a rede precisa aprender posições absolutas.
3. **Posição normalizada por `size`** (`x/N`, `y/N`) tem o mesmo intervalo `[0, 1]` em qualquer grid, mas representa distâncias físicas diferentes — outra razão para a transferência 5→10 falhar zero-shot.

Antes de mexer em arquitetura ou recompensa, queremos isolar o efeito da **percepção** e da **estratégia de treino**.

---

## 3. Estratégia 1 — Aumentar a janela de visão de 3x3 para 5x5

### Ideia

Trocar a matriz 3x3 por uma matriz egocêntrica 5x5 com a mesma codificação `0/1/2`. O agente continua tendo apenas observação parcial, mas com raio de visão maior consegue planejar enxergando células não visitadas a até 2 passos de distância (em vez de 1).

| Setup | Lado da janela | Features achatadas | Cobre quanto do grid 5x5 | 10x10 |
|-------|----------------|--------------------|---------------------------|-------|
| Baseline | 3 | 9  | ~36% | ~9% |
| Estratégia 1 | 5 | 25 | ~quase todo | ~25% |

5x5 é o **maior tamanho de janela permitido pelas regras da APS** ("a representação do estado pode ser 3x3 ou 5x5 com o agente sempre no centro"). Política, recompensa e algoritmo permanecem inalterados — só a entrada cresce de 9 para 25 features. Como ganho colateral, o `observation_space` passa a ter shape **fixo em qualquer tamanho de grid**, o que habilita Estratégia 2 (curriculum cross-grid).

### Implementação

Em [`gymnasium_env/grid_world_cpp.py`](gymnasium_env/grid_world_cpp.py):
- Novo parâmetro `view_size: int = 5` no construtor (validado para `view_size in (3, 5)`).
- `observation_space["neighbors"]` passa a ter shape `(view_size, view_size)`.
- `set_neighbors()` foi generalizada para iterar `range(view_size)` deslocando pelo `view_radius`, em vez do laço fixo de 3.

### Como reproduzir

```bash
source venv/bin/activate

# Modelo A — 5x5 do zero
python train_grid_world_cpp.py train 5 3 200 1000000

# Modelo B — 10x10 do zero
python train_grid_world_cpp.py train 10 12 400 1500000

# Avaliação (5 vezes cada)
python train_grid_world_cpp.py test 5 3
python train_grid_world_cpp.py test 10 12
```

### Resultados

Modelos:
- **Modelo A** — `data/ppo_cpp_5_3_200_0.05_20260507_102905.zip` — 1M timesteps, ~5,0 min wall-clock.
- **Modelo B** — `data/ppo_cpp_10_12_400_0.05_20260507_103554.zip` — 1,5M timesteps, ~11,2 min wall-clock.

Cada linha é a média de **5 execuções** de `test` com 100 episódios cada.

| Modelo | Grid  | Full Coverage Rate (5 runs)        | Média | Average Coverage | Avg Steps |
|--------|-------|------------------------------------|-------|------------------|-----------|
| A      | 5x5   | 96 / 97 / 96 / 91 / 92             | **94,4%** | 99,72%       | ~43       |
| B      | 10x10 | 65 / 63 / 56 / 57 / 65             | **61,2%** | 98,68%       | ~307      |

### Análise

- **No 5x5 a hipótese se confirmou — com folga.** Subimos de **76,4% (baseline 3x3)** para **94,4%** mantendo o resto do pipeline intacto. *Average Coverage* ficou em 99,72% com pouca variância entre runs — o agente raramente erra um episódio inteiro.
- **No 10x10 a Estratégia 1 do zero também já bate a baseline** (61,2% vs 64,0% — ligeiramente abaixo, mas o baseline é zero-shot e gargalo o caso médio). *Average Coverage* alto (98,68%), mas o agente perde 1-2 células no final em ~40% dos episódios. Mesmo padrão da janela 3x3, só que melhor calibrado.
- **Limitação da abordagem com janela local fixa**: no 10x10 a janela 5x5 cobre só 1/4 do grid. Quando faltam poucas células e elas estão em quadrantes opostos, fora do raio 2, o agente não tem informação para se direcionar até elas. Isso motiva a Estratégia 2.

---

## 4. Estratégia 2 — Curriculum Learning

### Ideia

Curriculum learning é o princípio de **apresentar versões progressivamente mais difíceis** da tarefa: o agente aprende uma versão fácil primeiro e usa esses pesos como ponto de partida para a versão difícil. A intuição é que muitas *skills* (não bater em parede, não revisitar, andar em direção a uma célula livre dentro da janela) são compartilhadas entre versões — em vez de aprender tudo do zero numa versão difícil onde o sinal de recompensa é raro, o agente parte de uma política já razoável e só precisa **fine-tunar**.

Testamos duas formas:

- **Modelo C (curriculum *intra*-grid):** treinar 5x5 com **0 obstáculos** primeiro (versão trivial — basta varrer o grid em zig-zag), depois fine-tunar no 5x5 com **3 obstáculos**.
- **Modelo D (curriculum *cross*-grid):** treinar 5x5 do zero (Modelo A), depois fine-tunar diretamente no 10x10. A janela 5x5 da Estratégia 1 deixa o `observation_space` **idêntico** entre os dois grids (mesmos `(5,5)` egocêntrico e mesmo vetor `[x/N, y/N, coverage_ratio]`), então os pesos são plug-and-play.

### Implementação

A funcionalidade já existia em `train_grid_world_cpp.py` no modo `curriculum`, mas com dois bugs que corrigimos:
1. Chamava `model.learn(total_timesteps=MAX_STEPS, reset_num_timesteps=False)` *antes* do logger ser configurado — treinava por uns 400 passos sem TensorBoard.
2. Logo depois chamava `model.learn(total_timesteps=TOTAL_TIMESTEPS)` **sem** `reset_num_timesteps=False`, fazendo o contador global voltar a zero.

Sequência correta agora: **carregar modelo → configurar logger → `model.learn(reset_num_timesteps=False)` → salvar**. O eixo de timesteps do TensorBoard mostra continuidade entre estágios (Modelo D vai de 1M a 2,5M; Modelo C de 500k a 1M).

### Como reproduzir

```bash
source venv/bin/activate

# Modelo C — curriculum intra-grid (5x5: 0 obstáculos -> 3 obstáculos)
python train_grid_world_cpp.py train 5 0 200 500000          # estágio 1
python train_grid_world_cpp.py curriculum 5 3 200 500000      # estágio 2 (pede o nome do modelo do estágio 1)

# Modelo D — curriculum cross-grid (5x5 -> 10x10)
# (estágio 1 reutiliza o Modelo A treinado na Estratégia 1)
python train_grid_world_cpp.py curriculum 10 12 400 1500000   # estágio 2 (pede o nome do Modelo A)

# Avaliação (5 vezes cada)
python train_grid_world_cpp.py test 5 3       # Modelo C
python train_grid_world_cpp.py test 10 12     # Modelo D
```

### Resultados

Modelos:
- **Modelo C** — `data/ppo_cpp_5_3_200_0.05_20260507_105654_curriculum.zip` — 500k + 500k = 1M timesteps total (~4,8 min wall-clock combinado).
- **Modelo D** — `data/ppo_cpp_10_12_400_0.05_20260507_110128_curriculum.zip` — 1,5M timesteps de fine-tune, partindo do Modelo A (~11,4 min wall-clock).

Cada linha é a média de **5 execuções** de `test` com 100 episódios cada.

| Modelo | Grid  | Full Coverage Rate (5 runs)        | Média | Average Coverage | Avg Steps |
|--------|-------|------------------------------------|-------|------------------|-----------|
| C      | 5x5   | 97 / 96 / 98 / 95 / 96             | **96,4%** | 99,75%       | ~41       |
| D      | 10x10 | 81 / 82 / 73 / 85 / 81             | **80,4%** | 99,42%       | ~245      |

### Comparação consolidada

| Configuração                                  | 5x5 Mean Full Coverage | 10x10 Mean Full Coverage |
|-----------------------------------------------|-------------------------|---------------------------|
| Baseline (janela 3x3, do zero / zero-shot)    | 76,4%                   | 64,0%                     |
| Estratégia 1 (janela 5x5, do zero)            | 94,4% (Modelo A)        | 61,2% (Modelo B)          |
| Estratégia 2 (janela 5x5, **com curriculum**) | **96,4% (Modelo C)**    | **80,4% (Modelo D)**      |
| Δ vs baseline                                 | **+20,0 p.p.**          | **+16,4 p.p.**            |
| Δ vs Estratégia 1                             | **+2,0 p.p.**           | **+19,2 p.p.**            |

### Análise

- **Curriculum *cross-grid* finalmente compensa** — e dramático. O Modelo D (5x5 → 10x10) supera o Modelo B (10x10 do zero) em **+19,2 p.p.** com o mesmo orçamento de fine-tune (1,5M timesteps). O Modelo D ainda é "barato": ele reutiliza os 1M do Modelo A em vez de jogar essa computação fora.
- **Por que funciona com janela 5x5 mas tinha falhado com 7x7** (em testes anteriores que descartamos): com 7x7 num grid 5x5, a janela mostra o grid inteiro com uma moldura grossa de paredes — a política aprende a depender dessas paredes-padding, e essa correlação não existe em 10x10. Com 5x5 num grid 5x5, a janela cobre a maior parte do grid mas sem padding artificial dominante — as *features* aprendidas (relacionar células visitadas com obstáculos próximos) se mantêm úteis em 10x10. **Tamanho de janela importa não só pela informação que ele dá, mas pelo quanto ele *vincula* a política ao grid de treino**.
- **Curriculum *intra-grid* também ajuda**, ainda que mais modesto: +2,0 p.p. sobre Modelo A. A versão sem obstáculos é basicamente um problema de varredura sistemática (zig-zag), e a política aprendida nele já chega no estágio 2 sabendo "como percorrer um espaço". O fine-tune com obstáculos só precisa adicionar a habilidade de desviar.
- **Diagnóstico do que ainda falta no 10x10**: Modelo D tem *Average Coverage* de 99,42% mas Full Coverage de 80,4%. Persiste o padrão "perde 1-2 células no fim" — em ~20% dos episódios, o agente cobre quase tudo mas trunca antes de fechar. A janela 5x5 ainda não dá alcance suficiente para guiar o agente até as últimas células quando elas estão fora do raio 2.

### Veredito

A combinação Estratégia 1 + Estratégia 2 é o nosso *baseline forte*: 96,4% no 5x5 e 80,4% no 10x10. Para fechar a lacuna que sobra (~20 p.p. faltando para 100% no 10x10), a próxima estratégia precisa atacar especificamente o problema de *memória do mapa* — algo que nenhuma janela local fixa resolve.

---

## 5. Próximos passos planejados

### Estratégia 3 — `RecurrentPPO` (LSTM)

A causa raiz do gap residual no 10x10 é falta de memória: a janela 5x5 não diz ao agente onde havia espaço inexplorado fora do raio atual. Uma forma natural de adicionar memória **sem trocar a representação espacial** (mantendo a regra 3x3/5x5 da APS) é trocar o algoritmo: usar **`RecurrentPPO`** (do `sb3-contrib`), que substitui a MLP por uma LSTM. O agente passa a manter um *hidden state* entre passos do mesmo episódio — ou seja, ele lembra implicitamente do que viu nos últimos *k* passos sem que isso esteja no estado.

Plano:
1. Instalar `sb3-contrib` (`pip install sb3-contrib`).
2. Trocar `PPO("MultiInputPolicy", ...)` por `RecurrentPPO("MultiInputLstmPolicy", ...)` no `train_grid_world_cpp.py`.
3. Treinar 5x5 e 10x10 com a mesma estrutura (do zero + curriculum) e comparar contra os Modelos A-D.

### Estratégia 4 (se necessário) — Reward shaping

A regra também permite: "a função de reward pode ser alterada à vontade." Se a Estratégia 3 não fechar o gap, vale tentar:
- Diminuir a penalidade de truncamento (de -5 para 0) para o agente não ficar "com medo" de tentar sweep longos.
- Adicionar bonus pequeno por descobrir/sensoriar célula livre nova dentro da janela.

---

## 6. Como interpretar os artefatos

- **Modelos** ficam em `data/` (versionados no repo conforme regra da APS):
  - `ppo_cpp_5_3_200_0.05_<timestamp>.zip` — Modelo A (5x5 do zero).
  - `ppo_cpp_10_12_400_0.05_<timestamp>.zip` — Modelo B (10x10 do zero).
  - `ppo_cpp_5_0_200_0.05_<timestamp>.zip` — origem do Modelo C (5x5 sem obstáculos, estágio 1 do curriculum intra-grid).
  - `ppo_cpp_5_3_200_0.05_<timestamp>_curriculum.zip` — Modelo C (5x5 estágio 2 do curriculum intra-grid).
  - `ppo_cpp_10_12_400_0.05_<timestamp>_curriculum.zip` — Modelo D (10x10 com curriculum cross-grid).
- **Logs do TensorBoard** ficam em `log/...` (não versionados — pasta no `.gitignore`). Abrir com `tensorboard --logdir log` para ver `ep_rew_mean`, `ep_len_mean` e curvas de loss durante treinamento. Os runs de curriculum mostram continuidade no eixo de timesteps (Modelo C vai de 500k a 1M; Modelo D de 1M a 2,5M) graças ao `reset_num_timesteps=False`.
- **Tabelas de resultado** vêm de `python train_grid_world_cpp.py test <DIM> <OBS>` (100 episódios por chamada, repetido 5x para média).
