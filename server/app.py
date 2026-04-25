# =============================================================
# server/app.py  —  SmartCity Traffic Control System
# HOW IT WORKS:
#   OpenEnv's create_fastapi_app automatically creates all the
#   standard endpoints (/reset, /step, /state, /health, /docs).
#   We just pass it our environment, action, and observation
#   classes — it handles the rest.
# =============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from openenv.core.env_server import create_fastapi_app

from models import TrafficAction, TrafficObservation
from smartcity_traffic_environment import CityTrafficEnvironment


# =============================================================
# CREATE THE APP — one line using OpenEnv's factory
# =============================================================

# create_fastapi_app takes:
#   1. env       — a CALLABLE (lambda/function) that creates env instances
#                  NOT the instance itself — OpenEnv manages instances
#   2. action_cls       — our TrafficAction class
#   3. observation_cls  — our TrafficObservation class

app = create_fastapi_app(
    env             = lambda: CityTrafficEnvironment(task="easy"),
    action_cls      = TrafficAction,
    observation_cls = TrafficObservation,
)

# =============================================================
# WHY LAMBDA?
# =============================================================
# create_fastapi_app expects a factory function — something it
# can CALL to create a new environment instance when needed.
#
# WRONG:  create_fastapi_app(env=CityTrafficEnvironment())
#         This creates ONE instance and passes it — but OpenEnv
#         wants to create instances itself for isolation.
#
# RIGHT:  create_fastapi_app(env=lambda: CityTrafficEnvironment())
#         This passes a function. OpenEnv calls it when needed.


# =============================================================
# RUN THE SERVER
# =============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  SmartCity Traffic Control — OpenEnv Server")
    print("=" * 55)
    print("  Endpoints auto-created by OpenEnv:")
    print("    POST /reset  — start new episode")
    print("    POST /step   — send one action, get observation")
    print("    GET  /state  — see full city state")
    print("    GET  /health — server alive check")
    print("    GET  /docs   — interactive API documentation")
    print("=" * 55)
    print("  Visit http://localhost:8000/docs to explore the API")
    print("=" * 55)

    uvicorn.run(
        "app:app",
        host   = "0.0.0.0",
        port   = 8000,
        reload = False,
    )
