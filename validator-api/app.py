import asyncio
import os
from datetime import datetime
import time
from typing import Annotated, List
import random

import bittensor
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from starlette import status
from substrateinterface import Keypair

from omega.protocol import Videos
from omega.imagebind_wrapper import ImageBind

from validator_api import score
from validator_api.config import TOPICS_LIST, IS_PROD
from validator_api.dataset_upload import dataset_uploader


NETWORK = "ws://127.0.0.1:9944" or os.environ["NETWORK"]
NETUID = "24" or int(os.environ["NETUID"])


security = HTTPBasic()
imagebind = ImageBind()


def get_hotkey(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    keypair = Keypair(ss58_address=credentials.username)

    if keypair.verify(credentials.username, credentials.password):
        return credentials.username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Signature mismatch",
    )


async def main():
    app = FastAPI()

    subtensor = bittensor.subtensor(network=NETWORK)
    metagraph: bittensor.metagraph = subtensor.metagraph(NETUID)

    async def resync_metagraph():
        while True:
            """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
            print("resync_metagraph()")

            # Sync the metagraph.
            metagraph.sync(subtensor=subtensor)

            await asyncio.sleep(90)

    @app.on_event("shutdown")
    async def shutdown_event():
        dataset_uploader.submit()

    @app.post("/api/validate")
    async def validate(
        videos: Videos,
        hotkey: Annotated[str, Depends(get_hotkey)],
    ) -> float:
        if hotkey not in metagraph.hotkeys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Valid hotkey required",
            )

        uid = metagraph.hotkeys.index(hotkey)

        if not metagraph.validator_permit[uid]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Validator permit required",
            )

        start_time = time.time()
        computed_score = await score.score_and_upload_videos(videos, imagebind)
        print(f"Returning score={computed_score} for validator={uid} in {time.time() - start_time:.2f}s")
        return computed_score

    if not IS_PROD:
        @app.get("/api/count_unique")
        async def count_unique(
            videos: Videos,
        ) -> str:
            nunique = await score.get_num_unique_videos(videos)
            return f"{nunique} out of {len(videos.video_metadata)} submitted videos are unique"

        @app.get("/api/check_score")
        async def check_score(
            videos: Videos,
        ) -> dict:
            detailed_score = await score.score_videos_for_testing(videos, imagebind)
            return detailed_score

    @app.get("/api/topic")
    async def get_topic() -> str:
        return random.choice(TOPICS_LIST)
    
    @app.get("/api/topics")
    async def get_topics() -> List[str]:
        return TOPICS_LIST

    @app.get("/")
    def healthcheck():
        return datetime.utcnow()

    await asyncio.gather(
        resync_metagraph(),
        asyncio.to_thread(uvicorn.run, app, host="0.0.0.0", port=8002)
    )


if __name__ == "__main__":
    asyncio.run(main())
