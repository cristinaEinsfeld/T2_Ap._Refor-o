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
        self.epsilon = max(
            self.hp["epsilon_min"],
            self.epsilon * self.hp["epsilon_decay"]
        )


# ============================================================================
# 4. LOOP DE TREINAMENTO
# ============================================================================
def train(hp: dict):
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
