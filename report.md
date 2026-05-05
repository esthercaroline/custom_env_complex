# Relatório — Coverage Path Planning com PPO

> Documento em construção. A cada iteração da implementação adicionamos uma nova seção com a estratégia testada, resultados e análise.

## 1. Contexto

O ambiente `GridWorldCPP` é uma adaptação do `GridWorld` para o problema de **Coverage Path Planning (CPP)**: o agente deve visitar todas as células livres de um grid `N x N` com obstáculos, com observação **parcial** do mapa.

- **Espaço de ações**: `Discrete(4)` — direita, cima, esquerda, baixo.
- **Espaço de observação original** (Dict, `MultiInputPolicy`):
  - `agent`: `[x/N, y/N, coverage_ratio]` (3 floats normalizados).
  - `neighbors`: matriz **3x3** centrada no agente, com `0` = livre não visitada, `1` = parede/obstáculo (incluindo fora do grid), `2` = visitada.
- **Função de recompensa**:
  - `+1.0` célula nova, `-0.3` revisita, `-0.5` colidir/ficar parado, `-0.1` por passo, `+10.0` cobertura completa, `-5.0` truncamento.
- **Algoritmo**: PPO (`stable-baselines3`) com `MultiInputPolicy`, `ent_coef=0.05`, `device="cpu"`.

### Baseline (configuração original do professor)

Hiperparâmetros: `DIM=5`, `OBSTACLES=3`, `MAX_STEPS=200`, `TOTAL_TIMESTEPS=1.000.000`.

| Grid | Full Coverage Rate (5 execuções de 100 episódios) | Média |
|------|----------------------------------------------------|-------|
| 5x5  | 75 / 78 / 69 / 79 / 81                             | ~76,4% |
| 10x10 (modelo treinado em 5x5) | 66 / 65 / 60 / 70 / 59       | ~64,0% |

**Observação**: o modelo treinado em 5x5 não generaliza bem para 10x10. Mesmo no próprio 5x5 o agente não atinge cobertura total em todos os episódios.

---

## 2. Diagnóstico

A informação espacial disponível ao agente é claramente o gargalo:

1. **Janela 3x3 é local demais.** Em um 5x5 essa janela cobre uma fração razoável do grid (≈36%), mas em 10x10 cobre apenas ≈9%. Quando o agente se afasta de regiões inexploradas, ele literalmente não consegue "ver" para onde ir — só percebe paredes, obstáculos e células visitadas imediatamente adjacentes.
2. **`MultiInputPolicy` achata a matriz 3x3 em 9 features** que entram em uma MLP padrão (2 camadas de 64 unidades). Não há *inductive bias* espacial; a rede precisa aprender posições absolutas.
3. **Posição normalizada por `size`** (`x/N`, `y/N`) tem o mesmo intervalo `[0, 1]` em qualquer grid, mas representa distâncias físicas diferentes — outra razão para a transferência 5→10 falhar.

Antes de mudar a arquitetura ou o algoritmo, queremos isolar o efeito da **percepção**: dar mais informação ao agente mantendo o restante do pipeline intacto.

---

## 3. Estratégia 1 — Aumentar a janela de visão local

### Ideia

Trocar a matriz 3x3 por uma matriz egocêntrica `view_size x view_size` (default **7x7**) com a mesma codificação `0/1/2`. O agente continua tendo apenas observação parcial (uma janela ao redor de si), mas com raio de visão maior consegue planejar o próximo passo enxergando células não visitadas a até 3 passos de distância.

| Setup | Lado da janela | Features achatadas | Cobre quanto do grid 5x5 | 10x10 |
|-------|----------------|--------------------|---------------------------|-------|
| Baseline | 3 | 9  | ~36% | ~9% |
| Estratégia 1 | 7 | 49 | ~quase todo | ~50% |

A política, a recompensa e o algoritmo permanecem **inalterados** — só a entrada cresce de 9 para 49 features.

### Implementação

Alterações em [`gymnasium_env/grid_world_cpp.py`](gymnasium_env/grid_world_cpp.py):

- Novo parâmetro `view_size: int = 7` no construtor (validado como ímpar e ≥ 3); `view_radius = view_size // 2`.
- `observation_space["neighbors"]` passa a ter shape `(view_size, view_size)`.
- `set_neighbors()` foi generalizada para iterar `range(view_size)` e deslocar pelos `view_radius`, em vez do laço fixo de 3.

Comandos para reproduzir:

```bash
source venv/bin/activate

# treinamento (mesmo orçamento da baseline)
python train_grid_world_cpp.py train 5 3 200 1000000
python train_grid_world_cpp.py train 10 12 400 1500000

# avaliação (100 episódios)
python train_grid_world_cpp.py test 5 3
python train_grid_world_cpp.py test 10 12
```

### Resultados (Estratégia 1, primeira execução)

| Grid  | Full Coverage Rate | Average Coverage | Std Dev | Min Cov | Max Cov | Avg Steps | Min/Max Steps |
|-------|--------------------|------------------|---------|---------|---------|-----------|---------------|
| 5x5   | **90 / 100**       | 98,45%           | 8,98%   | 13,64%  | 100,00% | 66,5      | 24 / 200      |
| 10x10 | **61 / 100**       | 98,91%           | 2,10%   | 87,50%  | 100,00% | 319,1     | 144 / 400     |

> Os números acima correspondem a uma única execução de teste (100 episódios) por grid; a baseline do professor reporta 5 execuções e por isso ainda precisamos rodar mais avaliações para comparar com rigor estatístico.

### Comparação com a baseline

| Grid  | Baseline (média de 5 runs) | Estratégia 1 (1 run) | Δ Full Coverage |
|-------|----------------------------|-----------------------|-----------------|
| 5x5   | ~76,4%                     | 90,0%                 | **+13,6 p.p.**  |
| 10x10 | ~64,0%                     | 61,0%                 | -3,0 p.p. (ruído ou regressão pequena) |

### Análise

- **No 5x5 a hipótese se confirmou.** Aumentar a janela de visão de 3x3 para 7x7 elevou a taxa de cobertura completa de ~76% para 90% mantendo tudo o mais constante. Em um grid pequeno, 7x7 essencialmente equivale a "ver o mapa inteiro local", o que torna a tarefa muito mais perto de plenamente observável.
- **No 10x10, a estória é mais interessante.** A *Average Coverage* subiu para **98,91%** com desvio-padrão de apenas 2,10% — ou seja, em quase todos os episódios o agente cobre quase tudo. Mas o *Full Coverage Rate* ficou em 61%, e o **número médio de passos foi 319/400** (80% do orçamento de tempo). O padrão sugere que **o agente cobre ~98% do grid mas se perde nas últimas 1–2 células e termina por *truncation***, não por má exploração.
- **Por que isso acontece?** Mesmo com janela 7x7, em um 10x10 ele só vê metade do grid. Quando faltam poucas células e elas estão em cantos opostos do mapa, fora do raio de 3, o agente não tem informação para se direcionar até elas. A política aprendida é boa em explorar o que vê, mas ruim em "lembrar" onde ainda há buracos.
- **Limitação clara da abordagem**: nenhuma janela local de tamanho fixo `< grid` resolve esse problema 100%; a partir de algum momento o agente *precisa* de uma representação do mapa parcial completo (que cresce com o grid), com algum mecanismo capaz de explorar essa estrutura espacial — tipicamente uma CNN.

### Veredito

A Estratégia 1 cumpre o critério de aceitação para 5x5 (≥ 95/100? Ainda não — 90/100 está perto) mas **não atinge o critério no 10x10** (61/100, alvo ≥ 90/100). Vamos avançar para a Estratégia 2.

---

## 4. Próximas estratégias planejadas

- **Estratégia 2 — Mapa global egocêntrico com padding + extrator CNN customizado.** Trocar `neighbors` por um tensor `(C, M, M)` com `M = 2·MAX_SIZE − 1` (e.g. 19), centrado no agente, padded com paredes fora do grid. Canais propostos: obstáculos *já vistos*, células visitadas, posição do agente. Mantém a observabilidade parcial usando uma estrutura `known_cells` que só revela obstáculos depois que entram no raio do sensor. Plugar um `BaseFeaturesExtractor` com 3 convoluções 3x3 + camada linear, conforme [a documentação do Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html).
- **Estratégia 3 — Tunar PPO** (`n_steps`, `batch_size`, schedule de `learning_rate`) e baixar `ent_coef` para 0,01 após a fase exploratória.
- **Estratégia 4 (se necessário)** — pequeno *bonus* de recompensa direcionado à célula livre mais próxima ainda dentro da janela visível, para amenizar o problema das "últimas células" observado no 10x10.

---

## 5. Como interpretar os artefatos

- Modelos treinados ficam em `data/ppo_cpp_<DIM>_<OBS>_<MAX_STEPS>_<ENT_COEF>_<timestamp>.zip`.
- Logs do TensorBoard em `log/ppo_cpp_<...>` — abrir com `tensorboard --logdir log` para ver `ep_rew_mean`, `ep_len_mean` e perdas durante o treinamento.
- Os números reportados na tabela vêm de `python train_grid_world_cpp.py test <DIM> <OBS>` (100 episódios por chamada).
