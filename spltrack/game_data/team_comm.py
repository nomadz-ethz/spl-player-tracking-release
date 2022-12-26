import json
from pathlib import Path


class TeamComm:
    def __init__(self, tc_logs):
        self._tc_logs = tc_logs

    @classmethod
    def read_from_json(cls, tc_logs_json_file_path: Path):
        with open(tc_logs_json_file_path, "r") as f:
            tc_logs = json.load(f)

        return cls(tc_logs)

    def __getitem__(self, idx):
        return self._tc_logs[idx]

    def __len__(self):
        return len(self._tc_logs)

    def get_fallen_events(self):
        fallen_events = []
        active_fallen_events_by_player = dict()
        for tc_msg in self._tc_logs:
            player_id = tc_msg["teamNum"], tc_msg["playerNum"]
            if tc_msg["fallen"]:
                if player_id not in active_fallen_events_by_player:
                    active_fallen_events_by_player[player_id] = {
                        "team_number": player_id[0],
                        "player_number": player_id[1],
                        "start_timestamp": tc_msg["timestamp"],
                        "end_timestamp": tc_msg["timestamp"],
                    }
                else:
                    active_fallen_events_by_player[player_id]["end_timestamp"] = tc_msg[
                        "timestamp"
                    ]
            else:
                if player_id in active_fallen_events_by_player:
                    fallen_event = active_fallen_events_by_player.pop(player_id)
                    fallen_events.append(fallen_event)
        return fallen_events
