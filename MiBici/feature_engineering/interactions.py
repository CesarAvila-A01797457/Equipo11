"""Cruces de categorías existentes como nuevas categorías."""

from MiBici.feature_engineering.base import FeatureStep


class InteractionStep(FeatureStep):
    """Crea ``a_x_b`` para cada par de columnas ya presentes en el frame."""

    def __init__(self, pairs):
        self.pairs = tuple(tuple(pair) for pair in pairs)
        self.categorical = tuple(f"{a}_x_{b}" for a, b in self.pairs)

    def transform(self, frame):
        frame = frame.copy()
        for (a, b), name in zip(self.pairs, self.categorical):
            missing = {a, b} - set(frame.columns)
            if missing:
                raise KeyError(f"InteractionStep requiere {sorted(missing)}")
            frame[name] = frame[a].astype("string") + "|" + frame[b].astype("string")
        return frame

    def __repr__(self):
        return f"InteractionStep({list(self.pairs)})"
