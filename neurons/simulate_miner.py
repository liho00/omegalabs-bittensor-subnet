# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import asyncio
import threading
import argparse
import traceback


class BaseMinerNeuron():
    """
    Base class for Bittensor miners.
    """

    def __init__(self, config=None):
        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None
        self.lock = asyncio.Lock()
        self.step = 0


    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """
        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                # Wait before checking again.
                time.sleep(1)

                # Check if we should exit.
                if self.should_exit:
                    break

                self.step += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            print(traceback.format_exc())

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            print("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            print("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            print("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            print("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()



import time
import typing
import bittensor as bt

# Bittensor Miner Template:
import omega

# from omega.base.miner import BaseMinerNeuron
from omega.imagebind_wrapper import ImageBind
from omega.miner_utils import search_and_embed_videos
from omega.augment import LocalLLMAugment, OpenAIAugment, NoAugment
from omega.utils.config import QueryAugment
from omega.constants import VALIDATOR_TIMEOUT

import torch


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        # query_augment_type = QueryAugment(self.config.neuron.query_augment)
        # if query_augment_type == QueryAugment.NoAugment:
        #     self.augment = NoAugment(device=self.config.neuron.device)
        # elif query_augment_type == QueryAugment.LocalLLMAugment:
        #     self.augment = LocalLLMAugment(device=self.config.neuron.device)
        # elif query_augment_type == QueryAugment.OpenAIAugment:
        #     self.augment = OpenAIAugment(device=self.config.neuron.device)
        # else:
        #     raise ValueError("Invalid query augment")
        self.imagebind = ImageBind()

    async def forward(
        self
    ) -> omega.protocol.Videos:
        query = "Explaining the complexities of global warming and climate change, including key factors and influential figures."
        num_videos = 8

        print(f"Received scraping request: {num_videos} videos for query '{query}'")
        start = time.time()
        # torch.cuda.set_device("cuda:0")
        video_metadata = search_and_embed_videos(
            query, num_videos, self.imagebind
        )
        time_elapsed = time.time() - start
        if len(video_metadata) == num_videos and time_elapsed < VALIDATOR_TIMEOUT:
            print(f"–––––– SCRAPING SUCCEEDED: Scraped {len(video_metadata)}/{num_videos} videos in {time_elapsed} seconds.")
        else:
            print(f"–––––– SCRAPING FAILED: Scraped {len(video_metadata)}/{num_videos} videos in {time_elapsed} seconds.")
        return None

    def save_state(self):
        """
        We define this function to avoid printing out the log message in the BaseNeuron class
        that says `save_state() not implemented`.
        """
        pass



if __name__ == "__main__":
    i = 0
    loop = asyncio.get_event_loop()
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
            if i == 0:
                loop.run_until_complete(miner.forward())
            i += 1
