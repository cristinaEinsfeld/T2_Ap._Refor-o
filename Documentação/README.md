# Agente Inteligente com Deep Q-Network (DQN) e Double DQN

**Trabalho 2 (T2) — Aprendizado por Reforço — PUCRS**
Escola Politécnica — Profª Daniela Oliveira Ferreira do Amaral

---

## Sumário

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Fundamentação Teórica e Artigos de Referência](#2-fundamentação-teórica-e-artigos-de-referência)
3. [O Ambiente: CartPole-v1](#3-o-ambiente-cartpole-v1)
4. [Arquitetura do Código — Explicação Parte a Parte](#4-arquitetura-do-código--explicação-parte-a-parte)
   - 4.1 [Hiperparâmetros](#41-hiperparâmetros)
   - 4.2 [QNetwork — Rede Neural Profunda](#42-qnetwork--rede-neural-profunda)
   - 4.3 [ReplayBuffer — Buffer de Experiência](#43-replaybuffer--buffer-de-experiência)
   - 4.4 [DQNAgent — O Agente Inteligente](#44-dqnagent--o-agente-inteligente)
   - 4.5 [Loop de Treinamento](#45-loop-de-treinamento)
   - 4.6 [Geração de Gráficos](#46-geração-de-gráficos)
   - 4.7 [Avaliação Visual](#47-avaliação-visual)
   - 4.8 [Ponto de Entrada Principal](#48-ponto-de-entrada-principal)
5. [Resultados Obtidos](#5-resultados-obtidos)
6. [Instalação e Execução](#6-instalação-e-execução)
7. [Estrutura de Arquivos](#7-estrutura-de-arquivos)
8. [Referências](#8-referências)

---

## 1. Visão Geral do Projeto

Neste projeto implementamos um agente baseado em **Deep Q-Network (DQN)** e na sua variante **Double DQN (DDQN)** para o ambiente `CartPole-v1` da biblioteca Gymnasium.

A ideia é que o agente aprenda, por tentativa e erro, a equilibrar um pêndulo invertido sobre um carrinho, tentando maximizar a recompensa acumulada ao longo dos episódios. O código foi escrito em **Python** usando **PyTorch** e organizado em seções para ficar mais fácil de entender.

---

## 2. Fundamentação Teórica e Artigos de Referência

Usamos três artigos científicos (Qualis A1-B1, publicados a partir de 2021) como base teórica:

| Código | Artigo | Contribuição para o projeto |
|--------|--------|-----------------------------|
| **[1]** | Zhang et al., *"On the Convergence and Sample Complexity Analysis of Deep Q-Networks with ε-Greedy Exploration"*, NeurIPS 2023. | Justifica matematicamente o decaimento do ε na política epsilon-greedy. Prova que ε alto no início amplia a região de convergência  e ε baixo acelera a taxa de convergência. Fundamenta a escolha do fator de desconto γ = 0.99. |
| **[2]** | Wang et al., *"Deep Reinforcement Learning: A Survey"*, IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 4, Abr. 2024. | Fundamenta a arquitetura MLP da Q-Network (Seção III-A), o uso do Replay Buffer para quebrar correlação temporal (Seção III-A, item 2), a Target Network para estabilidade (Seção III-A, item 3), e o algoritmo Double DQN (Seção III-B, Eq. 14). |
| **[3]** | Osei & Lopez, *"Experience Replay Optimisation via ATSC and TSC for Performance Stability in Deep RL"*, Appl. Sci. 2023, 13, 2034. | Embasa a escolha do tamanho do buffer (Seção 5, Figs. 5-6: buffers maiores produzem desempenho mais estável) e a estratégia de retenção FIFO implementada via `deque` (Seção 2.1). |

---

## 3. O Ambiente: CartPole-v1

O **CartPole-v1** é um problema clássico de controle: um pêndulo fica fixo sobre um carrinho que se move na horizontal, e o objetivo é manter o pêndulo em equilíbrio o maior tempo possível.

| Propriedade | Valor |
|-------------|-------|
| **Espaço de estados** | 4 dimensões contínuas: posição do carrinho, velocidade do carrinho, ângulo do pêndulo, velocidade angular |
| **Espaço de ações** | 2 ações discretas: empurrar para esquerda (0) ou para direita (1) |
| **Recompensa** | +1 a cada passo que o pêndulo permanece em pé |
| **Condição de término** | Ângulo > ±12° ou posição > ±2.4 do centro |
| **Truncamento** | Episódio truncado em 500 passos |
| **Meta de solução** | Média móvel >= 475 nos últimos 100 episódios |

---

## 4. Arquitetura do Código

O arquivo `dqn_cartpole.py` está organizado em 7 seções numeradas. Abaixo explicamos cada uma delas.

### 4.1 Hiperparâmetros


```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

HYPERPARAMS = {
    "env_name":           "CartPole-v1",
    "num_episodes":       1200,
    "lr":                 5e-4,
    "gamma":              0.99,
    "epsilon_start":      1.0,
    "epsilon_min":        0.01,
    "epsilon_decay":      0.995,
    "buffer_size":        100_000,
    "batch_size":         64,
    "target_update_freq": 10,
    "hidden_sizes":       (256, 256),
    "learn_every":        4,
    "use_double_dqn":     True,
    "use_soft_update":    False,
    "solve_threshold":    475,
    "loss_fn":            "huber",
    ...
}
```

**O que faz:** Junta todas as constantes do treinamento em um único dicionário, assim conseguimos ajustar qualquer parâmetro sem mexer na lógica do código.

**Justificativas teóricas:**

- **`gamma = 0.99`** — Zhang et al. [1] (Remark 4) mostram que a taxa de convergência é limitada por `γ + c_ε·(1−γ)`. Com γ próximo de 1, o agente valoriza recompensas futuras (horizonte longo), mas converge mais devagar. O valor 0.99 é o equilíbrio padrão para CartPole.

- **`epsilon_decay = 0.995`** — Implementa o decaimento geométrico prescrito por [1] (Corolário 1, Remark 2): o epsilon precisa diminuir à medida que os pesos W se aproximam do ótimo W*.

- **`buffer_size = 100_000`** — Wang et al. [2] (Seção III-A) explicam que o replay buffer quebra a correlação temporal dos dados. Osei & Lopez [3] (Seção 5, Figuras 5-6) demonstram experimentalmente que buffers maiores produzem desempenho mais estável.

- **`learn_every = 4`** — O agente só executa um passo de gradiente a cada 4 transições coletadas. Isso reduz o overfitting da rede neural (ela "vê" mais dados novos entre cada atualização) e melhora a estabilidade.

- **`loss_fn = "huber"`** — A Huber Loss (SmoothL1Loss no PyTorch) é menos sensível a outliers do que a MSE, o que estabiliza o gradiente quando o erro TD é grande.

---

### 4.2 QNetwork — Rede Neural Profunda

Classe `QNetwork`.

```
Entrada: estado s (4 dimensões: posição, velocidade, ângulo, vel. angular)
   |
Linear(4, 256) → ReLU
   |
Linear(256, 256) → ReLU
   |
Linear(256, 2)
   |
Saída: Q(s, esquerda), Q(s, direita)
```

**O que faz:** Recebe o vetor de estado de 4 dimensões do CartPole e retorna os Q-valores estimados para cada uma das 2 ações possíveis (empurrar esquerda ou direita).

**Como funciona internamente:**

1. O construtor (`__init__`) monta a rede dinamicamente a partir da tupla `hidden_sizes`. Para cada valor na tupla, adiciona uma camada `Linear` seguida de uma ativação `ReLU`. A última camada não tem ativação — ela produz os Q-valores brutos.

2. O método `forward` simplesmente passa o tensor de entrada pela rede sequencial e retorna a saída.

**Fundamentação teórica:**

- Wang et al. [2] (Seção III-A, item 1) descrevem que o DQN usa uma rede neural profunda para aproximar a função ação-valor Q(s,a). Hornik et al. (1989), citado em [2] Seção II-B, provam que MLPs multicamada são **aproximadores universais de funções**.

- A arquitetura segue a Equação (4) de Zhang et al. [1]: `H(W; x) = 1^T/K · φ(W_L^T ... φ(W_1^T x))`, onde `φ(.) = max{0, .}` (ReLU).

---

### 4.3 ReplayBuffer — Buffer de Experiência
 Classe `ReplayBuffer`.

**O que faz:** Armazena transições `(estado, ação, recompensa, próximo_estado, feito)` e fornece mini-batches aleatórios para o treinamento off-policy.

**Como funciona internamente:**

1. **Estrutura de dados:** Usa `collections.deque(maxlen=capacity)` — uma fila circular que automaticamente descarta as transições mais antigas quando atinge a capacidade máxima. Esta é a estratégia de retenção **FIFO** (First-In-First-Out).

2. **`push()`:** Encapsula os 5 valores em uma `namedtuple` chamada `Transition` e a adiciona à fila.

3. **`sample()`:** Usa `random.sample()` para extrair `batch_size` transições **uniformemente ao acaso**. Em seguida, desempacota as transições em 5 arrays NumPy separados, prontos para serem convertidos em tensores PyTorch.


---

### 4.4 DQNAgent — O Agente Inteligente

Classe `DQNAgent`. Contém 5 métodos:

#### 4.4.1 Construtor `__init__` 

**O que faz:** Inicializa todos os componentes do agente:
- Cria **duas redes neurais idênticas**: a rede local (`q_network`) e a rede alvo (`target_network`). A rede alvo começa com os mesmos pesos da local e fica em modo `eval()` (não calcula gradientes).
- Cria o otimizador **Adam** para atualizar os pesos da rede local.
- Escolhe a **função de perda** (Huber ou MSE).
- Cria o **ReplayBuffer**.
- Inicializa o **epsilon** em 1.0 (exploração total).

#### 4.4.2 `select_action(state)` 

**O que faz:** Implementa a política **ε-greedy**.

**Lógica:**
```
Se random() < epsilon:
    -> Retorna ação aleatória (EXPLORAÇÃO)
Senão:
    -> Converte estado para tensor
    -> Passa pela rede local (sem gradiente)
    -> Retorna argmax dos Q-valores (APROVEITAMENTO)
```

**Fundamentação:** Zhang et al. [1] (Algoritmo 1, linha 6): *"at state s_n, with probability ε_t, select a random action a_n, otherwise select a_n = argmax_a Q(W; s_n, a)."*

#### 4.4.3 `step(state, action, reward, next_state, done)`

**O que faz:** Recebe uma transição do ambiente, salva no buffer e decide se é hora de aprender.

**Lógica:**
```
1. Empurra a transição para o buffer
2. Incrementa contador de passos
3. Se (passo é múltiplo de learn_every) E (buffer tem dados suficientes):
      -> Chama self.learn()
   Senão:
      -> Retorna None (nenhum aprendizado neste passo)
```

O parâmetro `learn_every = 4` faz com que o agente colete 4 transições novas antes de cada atualização de gradiente. Isso dá à rede mais dados frescos entre cada ajuste de pesos, reduzindo overfitting.

#### 4.4.4 `learn()`  — **Método central do projeto**

**O que faz:** Executa um passo de otimização por gradiente descendente. É aqui que implementamos tanto o DQN padrão quanto o Double DQN.

**Passo a passo:**

```
1. Amostra mini-batch do buffer -> (states, actions, rewards, next_states, dones)
2. Converte tudo para tensores PyTorch no device correto
3. Calcula Q(s, a) pela rede LOCAL:
   q_values = q_network(states).gather(1, actions)
   -> gather seleciona apenas o Q-valor da ação que foi de fato tomada
4. Calcula o valor ALVO y (sem gradiente):

   SE Double DQN:
      a* = argmax_a' q_network(next_states)     <- rede LOCAL escolhe a ação
      y  = r + γ · target_network(s', a*)        <- rede ALVO avalia essa ação

   SE DQN padrão:
      y  = r + γ · max_a' target_network(s', a') <- rede ALVO faz tudo

   Em ambos os casos: y = 0 se o estado é terminal (multiplica por 1-done)

5. Calcula a perda: loss = HuberLoss(q_values, targets)
6. Retropropagação:
   - Zera gradientes
   - Calcula gradientes (loss.backward)
   - Clip de gradiente (norma máx. = 1.0) para estabilidade
   - Passo do otimizador (optimizer.step)
7. Retorna o valor escalar da perda
```

**Fundamentação da diferença DQN vs. DDQN:**

- **DQN padrão** [2, Eq. 13]: `y = R + γ · max_a Q-(s', a; θ-)`. A mesma rede (alvo) é usada para escolher E avaliar a melhor ação. Isso causa **superestimação** dos Q-valores: se a rede erra para cima em alguma ação, o `max` seleciona justamente esse erro inflado.

- **Double DQN** [2, Eq. 14]: `y = R + γ · Q-(s', argmax_a Q(s', a; θ); θ-)`. A rede **local** (θ) escolhe qual ação é a melhor; a rede **alvo** (θ-) avalia o valor dessa ação. Como as duas redes têm pesos diferentes, os erros tendem a não se reforçar, mitigando a superestimação.

#### 4.4.5 `update_target_network()`

**O que faz:** Sincroniza os pesos da target network com a rede local.

**Dois modos implementados:**

- **Hard update** (usado neste projeto): a cada 10 episódios, copia integralmente os pesos: `θ- <- θ`. Simples e eficaz.

- **Soft update** (Polyak averaging, disponível mas desabilitado): a cada passo, faz `θ- <- τ·θ + (1-τ)·θ-`. Uma mistura gradual entre as duas redes, originalmente do DDPG [2, Eq. 28].

**Fundamentação:** Wang et al. [2] (Seção III-A, item 3): *"The target network keeps the target value unchanged for some time, thereby enhancing the stability of DQN."*

#### 4.4.6 `decay_epsilon()`

**O que faz:** Aplica o decaimento geométrico `ε <- max(ε_min, ε * decay)` após cada episódio.

**Fundamentação:** Zhang et al. [1] (Corolário 1, Remark 2) provam que ε deve diminuir ao longo do treinamento:
- **ε alto no início** (Eq. 18): amplia a região de convergência, permitindo que o agente encontre o ótimo mesmo com uma inicialização ruim dos pesos.
- **ε baixo depois** (Eq. 19): a taxa de convergência é proporcional a `γ + c_ε·(1-γ)` — quanto menor c_ε (proporcional a ε), mais rápido o agente converge.
- Na prática, nos experimentos do artigo [1] (Seção 5, Atari Pong), os autores usam exatamente decaimento geométrico de 1.0 a 0.01.

---

### 4.5 Loop de Treinamento

Função `train(hp)`.

**Fluxo completo de um episódio:**

```
Para cada episódio (1 até 1200):
  1. Reseta o ambiente -> obtém estado inicial s0
  2. Para cada passo (até 500):
     a. Seleciona ação via ε-greedy
     b. Executa ação no ambiente -> recebe (s', r, done)
     c. Armazena transição no buffer
     d. A cada 4 passos, amostra mini-batch e atualiza rede
     e. (Se soft update) atualiza target a cada passo
     f. Se episódio terminou -> break
  3. (Se hard update) a cada 10 episódios, copia pesos para target
  4. Decai epsilon
  5. Registra métricas (score, loss, epsilon)
  6. Se média(100) > melhor média -> salva best_model.pth
  7. Se média(100) >= 475 -> imprime mensagem de sucesso
```

**Métricas monitoradas:**
- `scores[]`: recompensa acumulada de cada episódio
- `losses[]`: perda média de cada episódio
- `epsilons[]`: valor do epsilon ao fim de cada episódio
- `scores_window`: deque dos últimos 100 scores para média móvel

---

### 4.6 Geração de Gráficos

Função `plot_results()`.

**Gera três gráficos em uma única figura (salvos como `training_curves.png`):**

1. **Evolução da Recompensa Acumulada** — Mostra a recompensa bruta (barras azuis) e a média móvel de 100 episódios (linha laranja). Uma linha verde tracejada indica a meta de 475 pontos.

2. **Evolução da Perda (Loss)** — Mostra a Huber Loss média por episódio. A convergência da perda para valores baixos indica que a rede neural está aprendendo a prever os Q-valores corretamente.

3. **Decaimento do Epsilon** — Mostra a curva de ε de 1.0 até 0.01, evidenciando a transição de exploração (ε alto) para aproveitamento (ε baixo). Pode-se correlacionar visualmente: quando ε cai, a recompensa tende a subir.

---

### 4.7 Avaliação Visual

Função `evaluate_visual()`.

**O que faz:** Carrega um modelo `.pth` salvo, reconstrói a rede neural com a mesma arquitetura, e executa o agente no CartPole com renderização visual (`render_mode="human"`). Não há exploração — o agente sempre escolhe a ação com maior Q-valor (política puramente gulosa).

---

### 4.8 Ponto de Entrada Principal

Bloco `if __name__ == "__main__"`.

**O que faz:** Usa `argparse` para oferecer três modos de execução:

| Comando | Efeito |
|---------|--------|
| `python dqn_cartpole.py` | Treina com Double DQN (padrão) |
| `python dqn_cartpole.py --dqn` | Treina com DQN padrão (sem Double) |
| `python dqn_cartpole.py --eval` | Renderiza o agente usando `final_model.pth` |
| `python dqn_cartpole.py --eval --model best_model.pth` | Renderiza usando o melhor modelo |

---

## 5. Resultados Obtidos

Rodamos o treinamento com Double DQN por 1200 episódios e obtivemos:

| Métrica | Valor |
|---------|-------|
| Melhor média móvel (100 eps) | **455.68** |
| Pico de episódio individual | **500** (máximo do CartPole-v1) |
| Episódios com score = 500 | Múltiplos consecutivos (eps ~900-1100) |
| Loss final | ~0.06 (convergiu) |
| Tempo de treinamento (CPU) | ~4 minutos |

Os três gráficos gerados (`training_curves.png`) mostram:
- **Curva de recompensa ascendente** com média móvel atingindo ~450
- **Convergência da perda** para valores abaixo de 0.1
- **Decaimento suave do epsilon** correlacionado com a melhora na recompensa

---

## 6. Instalação e Execução

### Instalação

```bash
# Entrar no diretório do projeto
cd T2_Ap._Refor-o

# Criar ambiente virtual
python3 -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Instalar dependências
pip install gymnasium[classic-control] torch matplotlib numpy
```

### Execução

```bash
# Treinar com Double DQN (padrão)
python dqn_cartpole.py

# Treinar com DQN padrão (sem Double)
python dqn_cartpole.py --dqn

# Visualizar o agente treinado (melhor modelo)
python dqn_cartpole.py --eval --model best_model.pth

# Visualizar o agente treinado (modelo final)
python dqn_cartpole.py --eval
```


---

## 7. Referências

1. S. Zhang, H. Li, M. Wang, M. Liu, P.-Y. Chen, S. Lu, S. Liu, K. Murugesan, S. Chaudhury, *"On the Convergence and Sample Complexity Analysis of Deep Q-Networks with ε-Greedy Exploration"*, em Proc. 37th Conf. Neural Information Processing Systems (NeurIPS), 2023.

2. X. Wang, S. Wang, X. Liang, D. Zhao, J. Huang, X. Xu, B. Dai, Q. Miao, *"Deep Reinforcement Learning: A Survey"*, IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 4, pp. 5064-5078, Abr. 2024.

3. R. S. Osei, D. Lopez, *"Experience Replay Optimisation via ATSC and TSC for Performance Stability in Deep RL"*, Appl. Sci., vol. 13, no. 4, p. 2034, 2023.

4. V. Mnih et al., *"Human-level control through deep reinforcement learning"*, Nature, vol. 518, pp. 529-533, 2015.

5. H. Van Hasselt, A. Guez, D. Silver, *"Deep reinforcement learning with double Q-learning"*, em Proc. AAAI Conf. Artif. Intell., 2016.

6. R. S. Sutton, A. G. Barto, *Reinforcement Learning: An Introduction*, 2a ed. MIT Press, 2018.
