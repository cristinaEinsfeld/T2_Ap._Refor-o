#!/usr/bin/env python3
"""
==============================================================================
Trabalho 2 (T2) - Aprendizado por Reforço - PUCRS
Desenvolvimento de um Agente Inteligente com Deep Reinforcement Learning
==============================================================================

Agente DQN / Double DQN para o ambiente CartPole-v1 (Gymnasium).

Artigos de referência (Qualis A1-B1, pós-2021):
  [1] Zhang et al., "On the Convergence and Sample Complexity Analysis of
      Deep Q-Networks with epsilon-Greedy Exploration", NeurIPS 2023.
  [2] Wang et al., "Deep Reinforcement Learning: A Survey",
      IEEE Trans. Neural Netw. Learn. Syst., vol. 35, no. 4, Abr. 2024.
  [3] Osei & Lopez, "Experience Replay Optimisation via ATSC and TSC for
      Performance Stability in Deep RL", Appl. Sci. 2023, 13, 2034.

Autora: Cristina Einsfeld
Data  : Junho 2026
==============================================================================
"""

import random
import os
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Hiperparâmetros  (centralizados em dicionário para fácil ajuste)
# ---------------------------------------------------------------------------
# A taxa de aprendizado (lr) e o fator de desconto (gamma) são escolhidos com
# base na análise de convergência de Zhang et al. [1], Teorema 1: a taxa de
# convergência dos pesos W^{(t,0)} para W* é limitada superiormente por
# (gamma + c_eps * (1 - gamma)), onde gamma próximo de 1 enfatiza recompensas
# de longo prazo, porém exige mais iterações para convergir (Remark 4, [1]).
# Para o CartPole, gamma=0.99 equilibra horizonte longo e velocidade de
# convergência.
#
# O tamanho do buffer (buffer_size) segue a recomendação prática de
# Wang et al. [2] (Seção III-A, item 2): experience replay quebra a
# correlação temporal das amostras, e um buffer grande (>= 50 k) garante
# diversidade suficiente para estabilizar o gradiente.  Osei & Lopez [3]
# (Seção 5, Figuras 5-6) demonstram que buffers maiores produzem desempenho
# mais estável e com menor variância.
HYPERPARAMS = {
    "env_name":           "CartPole-v1",
    "num_episodes":       1200,
    "max_steps":          500,        # CartPole-v1 trunca em 500
    "lr":                 5e-4,       # [1] Seção 4.3 – tamanho do passo eta
    "gamma":              0.99,       # [1] Remark 4: gamma próximo de 1
    "epsilon_start":      1.0,        # [1] Corolário 1: eps alto no início
    "epsilon_min":        0.01,       # [1] Corolário 1: eps diminui com t
    "epsilon_decay":      0.995,      # decaimento geométrico de eps
    "buffer_size":        100_000,    # [2] Seção III-A; [3] Seção 5
    "batch_size":         64,         # tamanho do mini-batch padrão
    "target_update_freq": 10,         # hard update a cada N episódios
    "tau":                1e-2,       # coeficiente do soft update (Polyak)
    "hidden_sizes":       (256, 256), # camadas ocultas da MLP
    "learn_every":        4,          # aprende a cada N passos
    "use_double_dqn":     True,       # [2] Seção III-B (Double DQN)
    "use_soft_update":    False,      # False = hard update periódico
    "solve_threshold":    475,        # meta de média móvel (100 episódios)
    "loss_fn":            "huber",    # 'mse' ou 'huber'
}

# Diretório de saída para gráficos e modelo
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Transição (s, a, r, s', done)  –  tupla nomeada para clareza
# ---------------------------------------------------------------------------
Transition = namedtuple("Transition",
                        ("state", "action", "reward", "next_state", "done"))


# ============================================================================
# 1. REDE NEURAL PROFUNDA – QNetwork (MLP)
# ============================================================================
class QNetwork(nn.Module):
    """Aproximador da função Q usando uma MLP (Multi-Layer Perceptron).

    Fundamentação teórica ([2] Seção III-A, item 1):
        Wang et al. descrevem que o DQN utiliza uma rede neural profunda para
        extrair features de baixo nível e aproximar a função ação-valor Q(s,a)
        sem conhecimento de domínio adicional.  Hornik et al. (1989), citado
        em [2] Seção II-B, provam que MLPs multicamada são aproximadores
        universais de funções, justificando seu uso para representar Q*.

    A arquitetura segue a MLP com ReLU, conforme a Equação (4) de [1]:
        H(W; x) = 1^T / K * phi(W_L^T ... phi(W_1^T x))
    onde phi(.) = max{0, .} (ReLU).

    Argumentos:
        state_dim:    dimensão do espaço de estados (4 para CartPole).
        action_dim:   número de ações discretas (2 para CartPole).
        hidden_sizes: tupla com número de neurônios por camada oculta.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: tuple = (128, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna Q(s, a) para todas as ações dado o estado x."""
        return self.net(x)


# ============================================================================
# 2. REPLAY BUFFER (Buffer de Experiência)
# ============================================================================
class ReplayBuffer:
    """Buffer circular de experiência para treinamento off-policy.

    Fundamentação teórica ([2] Seção III-A, item 2; [3] Seções 2.1-2.2):
        Wang et al. [2] explicam que trajetórias consecutivas possuem
        correlação temporal; alimentar diretamente a rede com esses dados
        causa divergência entre valor estimado e esperado.  O experience
        replay armazena transições históricas e amostra uniformemente
        mini-batches aleatórios, quebrando a correlação e melhorando a
        eficiência de dados (treinamento off-policy).

        Osei & Lopez [3] (Seção 2.1) detalham a estratégia de retenção
        FIFO (First-In-First-Out), onde transições antigas são sobrescritas
        pelas novas quando o buffer está cheio.  Embora simples, esta
        abordagem é eficaz quando combinada com um buffer suficientemente
        grande (>= 50 k transições, cf. [3] Seção 5).

    Argumentos:
        capacity:   tamanho máximo do buffer (deque).
        batch_size: tamanho do mini-batch amostrado.
    """

    def __init__(self, capacity: int, batch_size: int):
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, done):
        """Armazena uma transição no buffer."""
        self.memory.append(Transition(state, action, reward,
                                      next_state, done))

    def sample(self):
        """Amostra um mini-batch aleatório uniforme.

        A amostragem uniforme (Pi = 1/n) é a estratégia padrão do DQN
        original ([2] Seção III-A; [3] Tabela 1).
        """
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.memory)


# ============================================================================
# 3. AGENTE DQN / DOUBLE DQN
# ============================================================================
class DQNAgent:
    """Agente Deep Q-Network com suporte a Double DQN.

    Fundamentação teórica:
        * DQN padrão – [2] Seção III-A, Eq. (12)-(13): utiliza target network
          para estabilizar o treinamento.  O valor alvo é:
              y = R + gamma * max_a Q^-(s', a; theta^-)
          onde Q^- é a target network com parâmetros theta^- atualizados
          periodicamente.

        * Double DQN – [2] Seção III-B, Eq. (14): desacopla seleção e
          avaliação da ação para mitigar superestimação dos Q-valores:
              y = R + gamma * Q^-(s', argmax_a Q(s', a; theta); theta^-)
          A rede local (theta) seleciona a melhor ação; a target (theta^-)
          avalia o valor dessa ação.  Van Hasselt et al. (2016) provaram que
          qualquer fonte de erro leva à superestimação no Q-learning padrão,
          e o DDQN corrige esse viés ([2] Seção III-B).

        * Política epsilon-greedy com decaimento – [1] Seção 4.1 (T1):
          Zhang et al. provam que epsilon deve decrescer ao longo das
          iterações para garantir convergência.  Um epsilon alto no início
          amplia a região de convergência (relaxa requisitos de W^{(0,0)}),
          enquanto um epsilon baixo acelera a taxa de convergência
          (Remark 5, [1]).  O decaimento geométrico epsilon_{t+1} =
          epsilon_t * decay implementa essa ideia diretamente (cf. Eq. 17
          e Remark 2 de [1]).

    Argumentos:
        state_dim:  dimensão do espaço de estados.
        action_dim: número de ações discretas.
        hp:         dicionário de hiperparâmetros.
    """

    def __init__(self, state_dim: int, action_dim: int, hp: dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hp = hp
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Rede local (Q) e rede alvo (Q^-)  –  [2] Seção III-A, item 3
        self.q_network = QNetwork(
            state_dim, action_dim, hp["hidden_sizes"]
        ).to(self.device)

        self.target_network = QNetwork(
            state_dim, action_dim, hp["hidden_sizes"]
        ).to(self.device)

        # Sincroniza pesos iniciais: theta^- = theta
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(),
                                   lr=hp["lr"])

        # Escolha da função de perda
        if hp["loss_fn"] == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

        # Buffer de experiência – [2] Seção III-A; [3] Seção 2
        self.buffer = ReplayBuffer(hp["buffer_size"], hp["batch_size"])

        # Epsilon corrente (inicia em epsilon_start)
        self.epsilon = hp["epsilon_start"]

        self.steps_done = 0

    # -----------------------------------------------------------------
    # Seleção de ação: epsilon-greedy com decaimento
    # -----------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        """Seleciona ação via política epsilon-greedy.

        Com probabilidade epsilon o agente explora (ação aleatória);
        com probabilidade (1 - epsilon) ele explota (argmax Q).

        Fundamentação ([1] Seção 3.1, Algoritmo 1, linha 6):
            "at state s_n, with probability epsilon_t, select a random
             action a_n, otherwise select a_n = argmax_a Q(W^{(t,0)}; s_n, a)."

        O decaimento é aplicado externamente após cada episódio (ver
        método decay_epsilon).
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return int(q_values.argmax(dim=1).item())

    # -----------------------------------------------------------------
    # Armazena transição e dispara aprendizado
    # -----------------------------------------------------------------
    def step(self, state, action, reward, next_state, done):
        """Salva transição no buffer e aprende a cada learn_every passos.

        Aprender a cada N passos (em vez de todo passo) reduz o número de
        atualizações de gradiente por experiência coletada, diminuindo o
        risco de overfitting e melhorando a estabilidade do treinamento.
        """
        self.buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1

        learn_every = self.hp.get("learn_every", 1)
        if (self.steps_done % learn_every == 0
                and len(self.buffer) >= self.hp["batch_size"]):
            return self.learn()
        return None

    # -----------------------------------------------------------------
    # Aprendizado (cálculo do Loss e retropropagação)
    # -----------------------------------------------------------------
    def learn(self) -> float:
        """Executa um passo de otimização sobre um mini-batch.

        Calcula a perda entre Q(s,a) estimado e o valor alvo y.

        Se use_double_dqn == False (DQN padrão, [2] Eq. 13):
            y = r + gamma * max_a' Q^-(s', a'; theta^-)

        Se use_double_dqn == True  (DDQN, [2] Eq. 14):
            a* = argmax_a' Q(s', a'; theta)        <-- rede local seleciona
            y  = r + gamma * Q^-(s', a*; theta^-)   <-- target avalia

        Retorna:
            Valor escalar da perda para registro de métricas.
        """
        states, actions, rewards, next_states, dones = self.buffer.sample()

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a) estimado pela rede local
        q_values = self.q_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            if self.hp["use_double_dqn"]:
                # ----- DOUBLE DQN ([2] Seção III-B, Eq. 14) -----
                # A rede local escolhe a melhor ação para s'
                best_actions = self.q_network(next_states_t).argmax(
                    dim=1, keepdim=True
                )
                # A target network avalia Q^-(s', a*)
                next_q = self.target_network(next_states_t).gather(
                    1, best_actions
                )
            else:
                # ----- DQN PADRÃO ([2] Seção III-A, Eq. 13) -----
                next_q = self.target_network(next_states_t).max(
                    dim=1, keepdim=True
                )[0]

            # Valor alvo: y = r + gamma * Q_target  (0 se estado terminal)
            targets = rewards_t + self.hp["gamma"] * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Clip de gradiente para estabilidade numérica
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # -----------------------------------------------------------------
    # Atualização da Target Network
    # -----------------------------------------------------------------
    def update_target_network(self):
        """Atualiza a target network.

        Fundamentação ([2] Seção III-A, item 3):
            "Every time the training completes a certain number of steps,
             the main network's parameters are synchronized to the target
             network."  Isso mantém o valor alvo estável por um período,
             melhorando a estabilidade do DQN.

        Dois modos disponíveis:
          * Hard update: theta^- <- theta  (cópia integral periódica)
          * Soft update (Polyak): theta^- <- tau*theta + (1-tau)*theta^-
            conforme [2] Seção IV-C, Eq. (28), originalmente do DDPG.
        """
        if self.hp["use_soft_update"]:
            tau = self.hp["tau"]
            for tp, lp in zip(self.target_network.parameters(),
                              self.q_network.parameters()):
                tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)
        else:
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )

    # -----------------------------------------------------------------
    # Decaimento do epsilon
    # -----------------------------------------------------------------
    def decay_epsilon(self):
        """Decaimento geométrico de epsilon: eps_{t+1} = eps_t * decay.

        Fundamentação ([1] Corolário 1, Remark 2):
            Zhang et al. provam que epsilon precisa diminuir conforme a
            distância e_t = ||W^{(t,0)} - W*|| decresce.  Um epsilon alto
            no início amplia a região de convergência (Eq. 18, [1]) mas
            desacelera a taxa; um epsilon baixo acelera a convergência
            (Eq. 19: taxa ~ gamma + c_eps*(1-gamma)).  O decaimento
            geométrico é a realização prática mais simples desta prescrição
            teórica (cf. Seção 5 de [1], onde eps_t decresce geometricamente
            de 1 a 0.01 nos experimentos com Atari Pong).
        """
        self.epsilon = max(
            self.hp["epsilon_min"],
            self.epsilon * self.hp["epsilon_decay"]
        )


# ============================================================================
# 4. LOOP DE TREINAMENTO
# ============================================================================
def train(hp: dict):
    """Treina o agente DQN/DDQN no ambiente especificado.

    Monitora recompensa acumulada, perda média e epsilon por episódio.
    Para quando atinge a meta de média móvel >= solve_threshold nos
    últimos 100 episódios ou quando esgota num_episodes.

    Fundamentação geral:
        O loop segue o Algoritmo 1 de [1] e o Algorithm 1 de [2]:
        para cada episódio (loop externo), o agente interage com o ambiente
        seguindo a política epsilon-greedy, armazena transições no replay
        buffer e, a cada passo, amostra um mini-batch para atualizar os
        pesos via gradiente descendente.

    Argumentos:
        hp: dicionário de hiperparâmetros.

    Retorna:
        Tupla (agent, scores, losses, epsilons).
    """
    env = gym.make(hp["env_name"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, hp)

    scores = []          # recompensa acumulada por episódio
    losses = []          # perda média por episódio
    epsilons = []        # epsilon no início de cada episódio
    scores_window = deque(maxlen=100)  # janela deslizante para média móvel

    mode_label = "Double DQN" if hp["use_double_dqn"] else "DQN"
    print(f"Iniciando treinamento ({mode_label}) no {hp['env_name']}")
    print(f"Dispositivo: {agent.device}")
    print("-" * 60)

    solved = False
    best_mean = -float("inf")

    for ep in range(1, hp["num_episodes"] + 1):
        state, _ = env.reset()
        score = 0.0
        ep_losses = []

        for _ in range(hp["max_steps"]):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                ep_losses.append(loss)

            # Soft update a cada passo, se habilitado
            if hp["use_soft_update"]:
                agent.update_target_network()

            state = next_state
            score += reward

            if done:
                break

        # Hard update periódico da target network
        if not hp["use_soft_update"]:
            if ep % hp["target_update_freq"] == 0:
                agent.update_target_network()

        # Decaimento do epsilon após cada episódio
        agent.decay_epsilon()

        # Registro de métricas
        scores.append(score)
        scores_window.append(score)
        avg_loss = np.mean(ep_losses) if ep_losses else 0.0
        losses.append(avg_loss)
        epsilons.append(agent.epsilon)

        mean_score = np.mean(scores_window)

        if ep % 50 == 0 or ep == 1:
            print(f"Ep {ep:4d} | Recompensa: {score:6.1f} | "
                  f"Média(100): {mean_score:7.2f} | "
                  f"Perda: {avg_loss:.4f} | Eps: {agent.epsilon:.4f}")

        # Salva o melhor modelo encontrado até agora
        if mean_score > best_mean:
            best_mean = mean_score
            best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(agent.q_network.state_dict(), best_path)

        if mean_score >= hp["solve_threshold"] and not solved:
            print(f"\n*** Ambiente resolvido no episódio {ep}! "
                  f"Média(100) = {mean_score:.2f} ***\n")
            solved = True

    env.close()

    # Salva pesos do modelo final
    model_path = os.path.join(OUTPUT_DIR, "final_model.pth")
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"Modelo final salvo em: {model_path}")
    print(f"Melhor modelo (média={best_mean:.2f}) salvo em: {best_path}")

    return agent, scores, losses, epsilons


# ============================================================================
# 5. GERAÇÃO DE GRÁFICOS
# ============================================================================
def plot_results(scores, losses, epsilons, hp):
    """Gera e salva três gráficos de evolução do treinamento.

    Fundamentação:
        A análise visual da curva de aprendizado é fundamental para validar
        que o agente de fato converge para Q* ([1] Seção 5, Figuras 1-3).
        Zhang et al. usam gráficos de test error vs. 1/sqrt(N), taxa de
        convergência vs. c_eps e test score vs. episódios para demonstrar
        alinhamento entre teoria e prática.
    """
    window = 100
    fig, axes = plt.subplots(3, 1, figsize=(10, 14))
    mode_label = "Double DQN" if hp["use_double_dqn"] else "DQN"

    # --- Gráfico 1: Recompensa Acumulada por Episódio ---
    ax1 = axes[0]
    ax1.plot(scores, alpha=0.4, color="steelblue", label="Recompensa")
    if len(scores) >= window:
        moving_avg = np.convolve(
            scores, np.ones(window) / window, mode="valid"
        )
        ax1.plot(
            range(window - 1, len(scores)),
            moving_avg, color="darkorange", linewidth=2,
            label=f"Média Móvel ({window} eps)"
        )
    ax1.axhline(y=hp["solve_threshold"], color="green",
                linestyle="--", label=f"Meta ({hp['solve_threshold']})")
    ax1.set_xlabel("Episódio")
    ax1.set_ylabel("Recompensa Acumulada")
    ax1.set_title(f"Evolução da Recompensa – {mode_label} ({hp['env_name']})")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # --- Gráfico 2: Perda Média por Episódio ---
    ax2 = axes[1]
    ax2.plot(losses, alpha=0.5, color="crimson", label="Perda Média")
    if len(losses) >= window:
        loss_avg = np.convolve(
            losses, np.ones(window) / window, mode="valid"
        )
        ax2.plot(
            range(window - 1, len(losses)),
            loss_avg, color="darkred", linewidth=2,
            label=f"Média Móvel ({window} eps)"
        )
    ax2.set_xlabel("Episódio")
    ax2.set_ylabel("Perda (Huber / MSE)")
    ax2.set_title(f"Evolução da Perda (Loss) – {mode_label}")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # --- Gráfico 3: Decaimento do Epsilon ---
    ax3 = axes[2]
    ax3.plot(epsilons, color="seagreen", linewidth=2,
             label="Epsilon (ε)")
    ax3.set_xlabel("Episódio")
    ax3.set_ylabel("Epsilon")
    ax3.set_title(
        "Decaimento do Epsilon – Exploração vs. Aproveitamento\n"
        "[1] Corolário 1: ε alto → região de convergência ampla; "
        "ε baixo → convergência rápida"
    )
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Gráficos salvos em: {fig_path}")


# ============================================================================
# 6. AVALIAÇÃO VISUAL (renderização do agente treinado)
# ============================================================================
def evaluate_visual(model_path: str, hp: dict, num_episodes: int = 3):
    """Carrega modelo salvo e renderiza o agente jogando CartPole.

    Argumentos:
        model_path:   caminho para o arquivo .pth com os pesos.
        hp:           dicionário de hiperparâmetros (para reconstruir a rede).
        num_episodes: número de episódios para renderizar.
    """
    env = gym.make(hp["env_name"], render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim, hp["hidden_sizes"])
    q_net.load_state_dict(torch.load(model_path, weights_only=True))
    q_net.eval()

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        score = 0.0
        done = False
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = int(q_net(state_t).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
        print(f"[Avaliação] Episódio {ep} – Recompensa: {score:.0f}")

    env.close()


# ============================================================================
# 7. PONTO DE ENTRADA PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DQN / Double DQN para CartPole-v1"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Modo avaliação: renderiza o agente treinado."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Caminho para o modelo .pth (modo avaliação)."
    )
    parser.add_argument(
        "--dqn", action="store_true",
        help="Usar DQN padrão em vez de Double DQN."
    )
    args = parser.parse_args()

    if args.dqn:
        HYPERPARAMS["use_double_dqn"] = False

    if args.eval:
        path = args.model or os.path.join(OUTPUT_DIR, "final_model.pth")
        evaluate_visual(path, HYPERPARAMS)
    else:
        agent, scores, losses, epsilons = train(HYPERPARAMS)
        plot_results(scores, losses, epsilons, HYPERPARAMS)
        print("\nTreinamento concluído.")
        print(f"Recompensa final (média dos últimos 100 episódios): "
              f"{np.mean(scores[-100:]):.2f}")
