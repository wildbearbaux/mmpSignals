import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Utilidades ────────────────────────────────────────────────────────────────

def plot_loss_curve(loss_curve: list, title: str = "Curva de pérdida") -> None:
    delta = [0] + [abs(loss_curve[i] - loss_curve[i - 1]) for i in range(1, len(loss_curve))]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(loss_curve, color="blue", label="Loss")
    ax1.set_xlabel("Iteración"); ax1.set_ylabel("Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(delta, color="red", alpha=0.5, label="ΔLoss")
    ax2.set_ylabel("ΔLoss", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(title)
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


def plot_confusion_matrix(y_true, y_pred, title: str = "Matriz de confusión") -> None:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 12})
    plt.title(title); plt.ylabel("Real"); plt.xlabel("Predicho")
    plt.tight_layout(); plt.show()


def remove_redundancy_by_corr_threshold(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return df.drop(columns=to_drop)


def evaluate(label: str, model, X_eval, y_eval, cv: StratifiedKFold) -> None:
    """Imprime accuracy con CV y reporte completo."""
    scores = cross_val_score(model, X_eval, y_eval, cv=cv, scoring="accuracy")
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Accuracy CV  : {scores.mean():.4f} ± {scores.std():.4f}")

    # Reentrenar para reporte final
    model.fit(X_eval, y_eval)
    y_pred = model.predict(X_eval)
    print(classification_report(y_eval, y_pred))
    plot_confusion_matrix(y_eval, y_pred, title=label)


# ── Carga y preprocesamiento ──────────────────────────────────────────────────

data = pd.read_excel("DataSet_completo.xlsx").drop(columns=["Longitud"])
feature_cols = [c for c in data.columns if c != "Clase"]
X_raw = data[feature_cols]
y = data["Clase"]

# Split fijo 80/20 (holdout) con estratificación para comparar de forma justa
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

# ── Arquitecturas a comparar ──────────────────────────────────────────────────

architectures = {
    "MLP (19,12)":   (19, 12),       # original
    "MLP (32,16)":   (32, 16),       # más capacidad
    "MLP (64,32,16)": (64, 32, 16),  # 3 capas
    "MLP (22,)":     (22,),          # compacta
}

def make_mlp(hidden_layers: tuple) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,   # ← más estable que 0.1
        max_iter=5_000,             # ← suficiente para converger
        early_stopping=True,        # ← detiene si val_loss no mejora
        validation_fraction=0.1,
        n_iter_no_change=30,        # ← paciencia
        random_state=42,
    )

# Pipeline con escalado interno (evita data leakage en CV)
def make_pipeline(hidden_layers: tuple) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    make_mlp(hidden_layers)),
    ])

# ── Evaluación con CV estratificado ──────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n── Todas las features (CV solo en train 80%) ───────────────────")
for name, layers in architectures.items():
    pipe = make_pipeline(layers)
    scores = cross_val_score(pipe, X_train_raw, y_train, cv=cv, scoring="accuracy")
    results[name] = scores.mean()
    print(f"  {name:20s}  {scores.mean():.4f} ± {scores.std():.4f}")

best_name = max(results, key=results.get)
best_layers = architectures[best_name]
print(f"\n  ✔ Mejor arquitectura: {best_name} ({results[best_name]:.4f})")

# ── Modelo final con todas las features ──────────────────────────────────────

best_pipe = make_pipeline(best_layers)
best_pipe.fit(X_train_raw, y_train)
plot_loss_curve(best_pipe.named_steps["mlp"].loss_curve_,
                title=f"Curva de pérdida — {best_name} (todas las features)")

y_test_pred = best_pipe.predict(X_test_raw)
print(f"\n  Accuracy holdout 20% (todas las features): {accuracy_score(y_test, y_test_pred):.4f}")
print(classification_report(y_test, y_test_pred))
plot_confusion_matrix(y_test, y_test_pred, title=f"Confusión — {best_name} (todas las features)")

# ── Modelo con features reducidas ────────────────────────────────────────────

X_reduced = remove_redundancy_by_corr_threshold(X_raw, threshold=0.85)
dropped = set(X_raw.columns) - set(X_reduced.columns)
print(f"\n── Features eliminadas por correlación alta: {dropped}")

X_train_reduced = X_train_raw[X_reduced.columns]
X_test_reduced = X_test_raw[X_reduced.columns]

results_reduced = {}
for name, layers in architectures.items():
    pipe = make_pipeline(layers)
    scores = cross_val_score(pipe, X_train_reduced, y_train, cv=cv, scoring="accuracy")
    results_reduced[name] = scores.mean()
    print(f"  {name:20s}  {scores.mean():.4f} ± {scores.std():.4f}")

best_name_r = max(results_reduced, key=results_reduced.get)
best_layers_r = architectures[best_name_r]
print(f"\n  ✔ Mejor con features reducidas: {best_name_r} ({results_reduced[best_name_r]:.4f})")

best_pipe_r = make_pipeline(best_layers_r)
best_pipe_r.fit(X_train_reduced, y_train)
plot_loss_curve(best_pipe_r.named_steps["mlp"].loss_curve_,
                title=f"Curva de pérdida — {best_name_r} (features reducidas)")

y_pred_r = best_pipe_r.predict(X_test_reduced)
print(f"\n  Accuracy holdout 20% (features reducidas): {accuracy_score(y_test, y_pred_r):.4f}")
print(classification_report(y_test, y_pred_r))
plot_confusion_matrix(y_test, y_pred_r, title=f"Confusión — {best_name_r} (features reducidas)")
