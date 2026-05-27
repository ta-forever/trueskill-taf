"""Load and filter the TAF game_stats.csv into the numpy arrays pytafskill expects.

The raw CSV has one row per (game, player); we group by gameId, decide team
membership from VICTORY vs DEFEAT, drop draws / unknowns / invalid games, and
emit arrays sorted by start time.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_CSV = Path(r"D:/wrk/trueskillcpp/data/TAF game_stats 20260313/game_stats.csv-1773390717847.csv")


def _parse_time_days(s: str) -> float:
    """Parse a 'YYYY-MM-DD HH:MM:SS' field (UTC assumed) as days since Unix epoch."""
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return dt.timestamp() / 86400.0


@dataclass
class GameRow:
    game_id: int
    mod: str
    start_days: float
    winners: list[str]   # playerNames who got VICTORY
    losers: list[str]    # playerNames who got DEFEAT


@dataclass
class Dataset:
    """Filtered, time-sorted games packed into pytafskill-shaped arrays."""

    pid1: np.ndarray            # (nGames, maxTeamSz) int32 — winners, -1 pad
    pid2: np.ndarray            # (nGames, maxTeamSz) int32 — losers,  -1 pad
    score: np.ndarray           # (nGames,) int32, always +1 (pid1 = winners)
    tstart: np.ndarray          # (nGames,) float64, days
    game_class: np.ndarray      # (nGames,) int32, always 0
    team_sizes: np.ndarray      # (nGames,) int32
    player_names: list[str]     # length nPlayers, index = pid
    n_players: int


def load_games(csv_path: Path = DEFAULT_CSV) -> list[GameRow]:
    """Parse the raw CSV into a list of GameRow, one per game, ignoring per-row
    pre/post ratings (we recompute those)."""
    by_game: dict[int, dict] = {}
    with csv_path.open(encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                gid = int(row["gameId"])
                validity = int(row["validity"])
            except ValueError:
                continue
            if validity != 0:
                continue
            g = by_game.get(gid)
            if g is None:
                g = by_game[gid] = {
                    "mod": row["mod"],
                    "start": row["startTime"],
                    "winners": [],
                    "losers": [],
                    "rejected": False,
                }
            result = row["result"]
            name = row["playerName"]
            if result == "VICTORY":
                g["winners"].append(name)
            elif result == "DEFEAT":
                g["losers"].append(name)
            else:
                # DRAW / UNKNOWN / CONFLICTING — mark this game as rejected
                g["rejected"] = True

    out: list[GameRow] = []
    for gid, g in by_game.items():
        if g["rejected"]:
            continue
        try:
            t = _parse_time_days(g["start"])
        except (ValueError, KeyError):
            continue
        if not g["winners"] or not g["losers"]:
            continue
        out.append(GameRow(
            game_id=gid,
            mod=g["mod"],
            start_days=t,
            winners=g["winners"],
            losers=g["losers"]))
    return out


def filter_games(games: Iterable[GameRow], *, mod: str, team_size: int | None = None,
                 min_team_size: int | None = None, max_team_size: int | None = None,
                 symmetric: bool = True) -> list[GameRow]:
    out: list[GameRow] = []
    for g in games:
        if g.mod != mod:
            continue
        w, l = len(g.winners), len(g.losers)
        if symmetric and w != l:
            continue
        ts = w  # team size after symmetry check
        if team_size is not None and ts != team_size:
            continue
        if min_team_size is not None and ts < min_team_size:
            continue
        if max_team_size is not None and ts > max_team_size:
            continue
        out.append(g)
    return out


def build_dataset(games: Iterable[GameRow]) -> Dataset:
    games_sorted = sorted(games, key=lambda g: g.start_days)
    if not games_sorted:
        raise ValueError("no games after filtering")
    max_team = max(len(g.winners) for g in games_sorted)
    name_to_pid: dict[str, int] = {}

    def pid_for(name: str) -> int:
        p = name_to_pid.get(name)
        if p is None:
            p = len(name_to_pid)
            name_to_pid[name] = p
        return p

    n = len(games_sorted)
    pid1 = np.full((n, max_team), -1, dtype=np.int32)
    pid2 = np.full((n, max_team), -1, dtype=np.int32)
    tstart = np.empty(n, dtype=np.float64)
    score = np.ones(n, dtype=np.int32)
    game_class = np.zeros(n, dtype=np.int32)
    team_sizes = np.empty(n, dtype=np.int32)

    for i, g in enumerate(games_sorted):
        ts = len(g.winners)
        team_sizes[i] = ts
        tstart[i] = g.start_days
        for j, w in enumerate(g.winners):
            pid1[i, j] = pid_for(w)
        for j, l in enumerate(g.losers):
            pid2[i, j] = pid_for(l)

    n_players = len(name_to_pid)
    names = [""] * n_players
    for name, p in name_to_pid.items():
        names[p] = name

    return Dataset(
        pid1=pid1, pid2=pid2, score=score, tstart=tstart,
        game_class=game_class, team_sizes=team_sizes,
        player_names=names, n_players=n_players)


def summarise(ds: Dataset, label: str = "") -> str:
    days = ds.tstart.max() - ds.tstart.min() if len(ds.tstart) else 0
    sizes = ", ".join(f"{int(s)}v{int(s)}:{int((ds.team_sizes == s).sum())}"
                      for s in sorted(set(ds.team_sizes.tolist())))
    return (f"{label}: {ds.n_players} players, {len(ds.tstart)} games over "
            f"{days:.0f} days  [{sizes}]")


if __name__ == "__main__":
    import sys
    games = load_games()
    print(f"Total valid games (after dropping draws/invalid): {len(games)}")
    prota = filter_games(games, mod="ProTA", team_size=1)
    esca  = filter_games(games, mod="Escalation", min_team_size=3)
    ds_p = build_dataset(prota)
    ds_e = build_dataset(esca)
    print(summarise(ds_p, "ProTA 1v1"))
    print(summarise(ds_e, "Escalation N-vs-N (N>=3)"))
