module.exports = {
    apps: [
        {
            name: "sn24-lihoominer-default",
            script: "neurons/miner.py",
            args: "--netuid 24 --subtensor.chain_endpoint ws://127.0.0.1:9944 --subtensor.network local --wallet.name lihoominer --wallet.hotkey default --axon.port 10818 --axon.external_port 10818 --neuron.device cuda:0",
            watch: false,
        },
        // {
        //     name: "sn24-lihoominer-defaulttwo",
        //     script: "neurons/miner.py",
        //     args: "--netuid 24 --subtensor.chain_endpoint ws://127.0.0.1:9944 --subtensor.network local --wallet.name lihoominer --wallet.hotkey defaulttwo --axon.port 20819 --axon.external_port 20819 --neuron.device cuda:0",
        //     watch: false,
        // },
    ]
}