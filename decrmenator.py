from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_actions: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class RLDiscriminatorConfig:
    max_features: int = 5000
    test_size: float = 0.2
    random_state: int = 42
    num_epochs: int = 10
    lr_policy: float = 1e-3
    lr_ce: float = 1e-4
    gamma_baseline: float = 0.9


class RLDiscriminator:
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        policy: PolicyNet,
        device: Optional[torch.device] = None,
        config: Optional[RLDiscriminatorConfig] = None,
    ) -> None:
        self.vectorizer = vectorizer
        self.policy = policy
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.config = config or RLDiscriminatorConfig()
        self.ce_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr_ce)

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        text_col: str = "TEXT",
        label_col: str = "LABEL",
        config: Optional[RLDiscriminatorConfig] = None,
    ) -> "RLDiscriminator":
        cfg = config or RLDiscriminatorConfig()

        df = pd.read_csv(csv_path)
        df[label_col] = df[label_col].astype(str).str.strip()

        label_map_binary = {
            "ham": 0,
            "Ham": 0,
            "spam": 1,
            "Spam": 1,
            "Smishing": 1,
            "smishing": 1,
        }
        df["target"] = df[label_col].map(label_map_binary)
        before = len(df)
        df = df.dropna(subset=["target"])
        after = len(df)
        print(f"[RLDiscriminator] Dropped {before - after} rows with unknown labels.")

        X_text = df[text_col].astype(str)
        y = df["target"].astype(int).values

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text,
            y,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y,
        )

        print(f"[RLDiscriminator] Train size: {len(X_train_text)}, Test size: {len(X_test_text)}")

        vectorizer = TfidfVectorizer(
            max_features=cfg.max_features,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words="english",
        )

        X_train_tfidf = vectorizer.fit_transform(X_train_text)
        X_test_tfidf = vectorizer.transform(X_test_text)

        n_features = X_train_tfidf.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = PolicyNet(input_dim=n_features).to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr_policy)

        num_epochs = cfg.num_epochs
        baseline = 0.0
        gamma_baseline = cfg.gamma_baseline

        X_train_csr = X_train_tfidf
        y_train_np = np.asarray(y_train, dtype=np.int64)

        policy.train()
        for epoch in range(1, num_epochs + 1):
            indices = np.random.permutation(len(y_train_np))
            epoch_rewards = []
            epoch_losses = []

            for idx in indices:
                x_vec = X_train_csr[idx].toarray().astype(np.float32)
                x_tensor = torch.from_numpy(x_vec).to(device)
                y_true = torch.tensor([y_train_np[idx]], device=device)

                logits = policy(x_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()

                reward = 1.0 if int(action.item()) == int(y_true.item()) else 0.0

                baseline = gamma_baseline * baseline + (1.0 - gamma_baseline) * reward
                advantage = reward - baseline

                loss = -dist.log_prob(action) * advantage

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_rewards.append(reward)
                epoch_losses.append(float(loss.item()))

            print(
                f"[RLDiscriminator] Epoch [{epoch}/{num_epochs}] "
                f"avg reward: {np.mean(epoch_rewards):.4f}, "
                f"avg loss: {np.mean(epoch_losses):.4f}"
            )

        policy.eval()
        correct = 0
        total = len(y_test)
        with torch.no_grad():
            for i in range(total):
                x_vec = X_test_tfidf[i].toarray().astype(np.float32)
                x_tensor = torch.from_numpy(x_vec).to(device)
                logits = policy(x_tensor)
                probs = F.softmax(logits, dim=-1)
                pred = probs.argmax(dim=-1).item()
                if int(pred) == int(y_test[i]):
                    correct += 1

        acc = correct / total if total > 0 else 0.0
        print(f"[RLDiscriminator] Test accuracy: {acc:.4f}")

        return cls(vectorizer=vectorizer, policy=policy, device=device, config=cfg)

    def _vectorize(self, msg: str) -> torch.Tensor:
        X_vec = self.vectorizer.transform([msg])
        x = X_vec.toarray().astype(np.float32)
        return torch.from_numpy(x).to(self.device)

    def classify(self, msg: str) -> Dict[str, object]:
        self.policy.eval()
        with torch.no_grad():
            x_tensor = self._vectorize(msg)
            logits = self.policy(x_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()

        pred = int(probs.argmax())
        label = "ham" if pred == 0 else "malicious"

        return {
            "label": label,
            "label_id": pred,
            "probs": {
                "ham": float(probs[0]),
                "malicious": float(probs[1]),
            },
        }

    def predict_and_self_update(
        self,
        msg: str,
        do_learn: bool = True,
    ) -> Tuple[int, np.ndarray, Optional[float]]:
        x_tensor = self._vectorize(msg)

        self.policy.train()
        logits = self.policy(x_tensor)
        probs = F.softmax(logits, dim=-1)
        probs_np = probs.detach().cpu().numpy().flatten()

        pred_tensor = probs.argmax(dim=-1)
        pred_label = int(pred_tensor.item())

        loss_value: Optional[float] = None
        if do_learn:
            loss = F.cross_entropy(logits, pred_tensor)
            self.ce_optimizer.zero_grad()
            loss.backward()
            self.ce_optimizer.step()
            loss_value = float(loss.item())

        return pred_label, probs_np, loss_value