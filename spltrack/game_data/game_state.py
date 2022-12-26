import json
from pathlib import Path

from ..data.constants import TEAM_NUMBER_TO_COLORS

_VALID_PENALTY_EVENTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


class GameState:
    def __init__(self, g_logs):
        self._g_logs = g_logs

    @classmethod
    def read_from_json(cls, g_logs_json_file_path: Path):
        with open(g_logs_json_file_path, "r") as f:
            g_logs = json.load(f)

        return cls(g_logs)

    @staticmethod
    def get_penalized_players_info(g_msg):
        team_info = g_msg["team"]
        penalized_players_info = []
        for ti in team_info:
            players_info = ti["player"]
            for player_idx, pi in enumerate(players_info):
                if pi["penalty"] in _VALID_PENALTY_EVENTS:
                    penalized_players_info.append(
                        {
                            "timestamp": g_msg["timestamp"],
                            "team_number": ti["teamNumber"],
                            "player_number": player_idx + 1,
                            "penalty": pi["penalty"],
                            "secsTillUnpenalised": pi["secsTillUnpenalised"],
                        }
                    )
        return penalized_players_info

    def get_penalized_events(self):
        penalized_events = []
        active_penalized_events_by_player = dict()
        for gc_msg in self._g_logs:
            penalized_players = GameState.get_penalized_players_info(gc_msg)
            penalized_players_by_player = {
                (x["team_number"], x["player_number"]): x for x in penalized_players
            }
            for player_id, pp in penalized_players_by_player.items():
                if player_id in active_penalized_events_by_player:
                    active_penalized_events_by_player[player_id]["end_timestamp"] = pp[
                        "timestamp"
                    ]
                else:
                    active_penalized_events_by_player[player_id] = {
                        "team_number": player_id[0],
                        "player_number": player_id[1],
                        "start_timestamp": pp["timestamp"],
                        "penalty": pp["penalty"],
                        "end_timestamp": pp["timestamp"],
                    }

            finished_penalized_events = []
            for player_id, penalized_event in active_penalized_events_by_player.items():
                if player_id not in penalized_players_by_player:
                    finished_penalized_events.append(player_id)

            for player_id in finished_penalized_events:
                penalized_event = active_penalized_events_by_player.pop(player_id)
                penalized_events.append(penalized_event)

        return sorted(penalized_events, key=lambda x: x["start_timestamp"])

    def __getitem__(self, idx):
        return self._g_logs[idx]

    def __len__(self):
        return len(self._g_logs)

    def get_team_info(self):
        g_logs_data = self._g_logs[0]
        assert (
            len(g_logs_data["team"]) == 2
        ), "GameController logs must have exactly 2 teams"
        team_info = {}
        for i, side in enumerate(["home", "away"]):
            team = g_logs_data["team"][i]
            team_number = team["teamNumber"]
            team_color = team["teamColor"]
            valid_team_colors = TEAM_NUMBER_TO_COLORS[team["teamNumber"]]
            if team_color not in valid_team_colors:
                print(
                    f"Warning: the color of team {team_number} in the logs is invalid!"
                    " Replacing with default home color."
                )
                team_color = valid_team_colors[0]
            team_info[side] = {
                "team_number": team_number,
                # Some GC logs have wrong team color -- overriding with our own constants
                "team_color": team_color,
            }
        return team_info
