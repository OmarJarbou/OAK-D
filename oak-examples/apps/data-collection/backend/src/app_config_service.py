from typing import Any, Dict, Callable

from nn import NNState


class GetAppConfigService:
    """
    Read-only service that exports current configuration state to the frontend.

    Aggregates state from:
    - get_model_state() callable (classes, confidence threshold)
    - get_snap_conditions_config() callable (snapping configuration export)
    """

    def __init__(
        self,
        get_nn_state: Callable[[], NNState],
        get_snap_conditions_config: Callable[[], Dict[str, Any]],
    ):
        self._get_nn_state = get_nn_state
        self._get_snap_conditions_config = get_snap_conditions_config

    def handle(self, payload: Any = None) -> Dict[str, Any]:
        nn_state = self._get_nn_state()
        return {
            "ok": True,
            "data": {
                "classes": list(nn_state.current_classes),
                "confidence_threshold": float(nn_state.confidence_threshold),
                "snapping": self._get_snap_conditions_config(),
            },
        }
