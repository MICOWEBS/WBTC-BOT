from web3 import Web3
from web3.middleware import geth_poa_middleware
import json
import logging
import config

# ABI for PancakeSwap Router
ROUTER_ABI = json.loads('''
[
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "amountIn",
                "type": "uint256"
            },
            {
                "internalType": "address[]",
                "name": "path",
                "type": "address[]"
            }
        ],
        "name": "getAmountsOut",
        "outputs": [
            {
                "internalType": "uint256[]",
                "name": "amounts",
                "type": "uint256[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "amountOut",
                "type": "uint256"
            },
            {
                "internalType": "address[]",
                "name": "path",
                "type": "address[]"
            }
        ],
        "name": "getAmountsIn",
        "outputs": [
            {
                "internalType": "uint256[]",
                "name": "amounts",
                "type": "uint256[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "amountOutMin",
                "type": "uint256"
            },
            {
                "internalType": "address[]",
                "name": "path",
                "type": "address[]"
            },
            {
                "internalType": "address",
                "name": "to",
                "type": "address"
            },
            {
                "internalType": "uint256",
                "name": "deadline",
                "type": "uint256"
            }
        ],
        "name": "swapExactETHForTokens",
        "outputs": [
            {
                "internalType": "uint256[]",
                "name": "amounts",
                "type": "uint256[]"
            }
        ],
        "stateMutability": "payable",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "amountIn",
                "type": "uint256"
            },
            {
                "internalType": "uint256",
                "name": "amountOutMin",
                "type": "uint256"
            },
            {
                "internalType": "address[]",
                "name": "path",
                "type": "address[]"
            },
            {
                "internalType": "address",
                "name": "to",
                "type": "address"
            },
            {
                "internalType": "uint256",
                "name": "deadline",
                "type": "uint256"
            }
        ],
        "name": "swapExactTokensForETH",
        "outputs": [
            {
                "internalType": "uint256[]",
                "name": "amounts",
                "type": "uint256[]"
            }
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "amountIn",
                "type": "uint256"
            },
            {
                "internalType": "uint256",
                "name": "amountOutMin",
                "type": "uint256"
            },
            {
                "internalType": "address[]",
                "name": "path",
                "type": "address[]"
            },
            {
                "internalType": "address",
                "name": "to",
                "type": "address"
            },
            {
                "internalType": "uint256",
                "name": "deadline",
                "type": "uint256"
            }
        ],
        "name": "swapExactTokensForTokens",
        "outputs": [
            {
                "internalType": "uint256[]",
                "name": "amounts",
                "type": "uint256[]"
            }
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]
''')

# ABI for ERC20 Token
TOKEN_ABI = json.loads('''
[
    {
        "constant": true,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": true,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": true,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": true,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": true,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {"name": "_from", "type": "address"},
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"}
        ],
        "name": "transferFrom",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    }
]
''')

class DexClient:
    def __init__(self, node_url, wallet_address, private_key):
        self.logger = logging.getLogger(__name__)
        self.web3 = Web3(Web3.HTTPProvider(node_url))
        self.web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self.wallet_address = wallet_address
        self.private_key = private_key
        
        # Initialize contracts
        self.router = self.web3.eth.contract(
            address=self.web3.to_checksum_address(config.PANCAKESWAP_ROUTER_ADDRESS),
            abi=ROUTER_ABI
        )
        
        self.wbtc = self.web3.eth.contract(
            address=self.web3.to_checksum_address(config.WBTC_ADDRESS),
            abi=TOKEN_ABI
        )
        
        self.wbnb = self.web3.eth.contract(
            address=self.web3.to_checksum_address(config.WBNB_ADDRESS),
            abi=TOKEN_ABI
        )
        
        self.busd = self.web3.eth.contract(
            address=self.web3.to_checksum_address(config.BUSD_ADDRESS),
            abi=TOKEN_ABI
        )
        
        # Get token decimals
        self.wbtc_decimals = self.wbtc.functions.decimals().call()
        self.wbnb_decimals = self.wbnb.functions.decimals().call()
        self.busd_decimals = self.busd.functions.decimals().call()
        
    def get_wbtc_price_in_busd(self):
        """Get the current price of WBTC in BUSD on PancakeSwap."""
        try:
            # Amount in with 18 decimals (1 WBTC)
            amount_in = 10 ** self.wbtc_decimals
            
            # Path from WBTC to BUSD
            path = [
                self.web3.to_checksum_address(config.WBTC_ADDRESS),
                self.web3.to_checksum_address(config.BUSD_ADDRESS)
            ]
            
            # Get amounts out
            amounts_out = self.router.functions.getAmountsOut(amount_in, path).call()
            
            # Calculate price
            wbtc_price = amounts_out[1] / (10 ** self.busd_decimals)
            
            return wbtc_price
        except Exception as e:
            self.logger.error(f"Error getting WBTC price: {e}")
            return None
            
    def check_liquidity(self, token_address):
        """Check if there's sufficient liquidity for trading."""
        try:
            # Path from token to WBNB
            path = [
                self.web3.to_checksum_address(token_address),
                self.web3.to_checksum_address(config.WBNB_ADDRESS)
            ]
            
            # Amount in with token decimals (1 token)
            token_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(token_address),
                abi=TOKEN_ABI
            )
            token_decimals = token_contract.functions.decimals().call()
            amount_in = 10 ** token_decimals
            
            # Try getting amounts out to check liquidity
            amounts_out = self.router.functions.getAmountsOut(amount_in, path).call()
            liquidity_in_bnb = amounts_out[1] / (10 ** self.wbnb_decimals)
            
            # Check if liquidity is sufficient (arbitrary threshold for illustration)
            sufficient_liquidity = liquidity_in_bnb > 0.1
            
            return {
                'sufficient': sufficient_liquidity,
                'liquidity_in_bnb': liquidity_in_bnb
            }
        except Exception as e:
            self.logger.error(f"Error checking liquidity: {e}")
            return {'sufficient': False, 'liquidity_in_bnb': 0}
    
    def get_wallet_balances(self):
        """Get wallet balances for relevant tokens."""
        try:
            wbtc_balance = self.wbtc.functions.balanceOf(self.wallet_address).call() / (10 ** self.wbtc_decimals)
            bnb_balance = self.web3.eth.get_balance(self.wallet_address) / (10 ** 18)
            busd_balance = self.busd.functions.balanceOf(self.wallet_address).call() / (10 ** self.busd_decimals)
            
            return {
                'wbtc': wbtc_balance,
                'bnb': bnb_balance,
                'busd': busd_balance
            }
        except Exception as e:
            self.logger.error(f"Error getting wallet balances: {e}")
            return None
    
    def approve_token(self, token_address, amount=None):
        """Approve router to spend tokens."""
        try:
            token_contract = self.web3.eth.contract(
                address=self.web3.to_checksum_address(token_address),
                abi=TOKEN_ABI
            )
            
            token_decimals = token_contract.functions.decimals().call()
            
            # If amount is None, approve max uint256
            if amount is None:
                amount = 2**256 - 1
            else:
                amount = int(amount * (10 ** token_decimals))
            
            # Build transaction
            tx = token_contract.functions.approve(
                config.PANCAKESWAP_ROUTER_ADDRESS,
                amount
            ).build_transaction({
                'from': self.wallet_address,
                'nonce': self.web3.eth.get_transaction_count(self.wallet_address),
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return receipt
        except Exception as e:
            self.logger.error(f"Error approving token: {e}")
            return None
    
    def buy_wbtc_with_busd(self, busd_amount, slippage_percent=config.MAX_SLIPPAGE_PERCENT):
        """Buy WBTC with BUSD."""
        if config.SIMULATION_MODE:
            self.logger.info(f"SIMULATION: Would buy WBTC with {busd_amount} BUSD")
            return {
                'success': True,
                'tx_hash': None,
                'simulation': True
            }
            
        try:
            # Calculate amount in with BUSD decimals
            amount_in = int(busd_amount * (10 ** self.busd_decimals))
            
            # Path from BUSD to WBTC
            path = [
                self.web3.to_checksum_address(config.BUSD_ADDRESS),
                self.web3.to_checksum_address(config.WBTC_ADDRESS)
            ]
            
            # Get expected amount out
            amounts_out = self.router.functions.getAmountsOut(amount_in, path).call()
            min_amount_out = int(amounts_out[1] * (1 - slippage_percent / 100))
            
            # Set deadline 20 minutes in the future
            deadline = self.web3.eth.get_block('latest').timestamp + 1200
            
            # Build transaction
            tx = self.router.functions.swapExactTokensForTokens(
                amount_in,
                min_amount_out,
                path,
                self.wallet_address,
                deadline
            ).build_transaction({
                'from': self.wallet_address,
                'nonce': self.web3.eth.get_transaction_count(self.wallet_address),
                'gas': 300000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                'success': receipt.status == 1,
                'tx_hash': tx_hash.hex(),
                'receipt': receipt
            }
        except Exception as e:
            self.logger.error(f"Error buying WBTC: {e}")
            return {'success': False, 'error': str(e)}
    
    def sell_wbtc_for_busd(self, wbtc_amount, slippage_percent=config.MAX_SLIPPAGE_PERCENT):
        """Sell WBTC for BUSD."""
        if config.SIMULATION_MODE:
            self.logger.info(f"SIMULATION: Would sell {wbtc_amount} WBTC for BUSD")
            return {
                'success': True,
                'tx_hash': None,
                'simulation': True
            }
            
        try:
            # Calculate amount in with WBTC decimals
            amount_in = int(wbtc_amount * (10 ** self.wbtc_decimals))
            
            # Path from WBTC to BUSD
            path = [
                self.web3.to_checksum_address(config.WBTC_ADDRESS),
                self.web3.to_checksum_address(config.BUSD_ADDRESS)
            ]
            
            # Get expected amount out
            amounts_out = self.router.functions.getAmountsOut(amount_in, path).call()
            min_amount_out = int(amounts_out[1] * (1 - slippage_percent / 100))
            
            # Set deadline 20 minutes in the future
            deadline = self.web3.eth.get_block('latest').timestamp + 1200
            
            # Build transaction
            tx = self.router.functions.swapExactTokensForTokens(
                amount_in,
                min_amount_out,
                path,
                self.wallet_address,
                deadline
            ).build_transaction({
                'from': self.wallet_address,
                'nonce': self.web3.eth.get_transaction_count(self.wallet_address),
                'gas': 300000,
                'gasPrice': self.web3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                'success': receipt.status == 1,
                'tx_hash': tx_hash.hex(),
                'receipt': receipt
            }
        except Exception as e:
            self.logger.error(f"Error selling WBTC: {e}")
            return {'success': False, 'error': str(e)}
    
    def estimate_gas_price(self):
        """Estimate current gas price on BSC."""
        try:
            gas_price = self.web3.eth.gas_price
            gas_price_gwei = self.web3.from_wei(gas_price, 'gwei')
            return gas_price_gwei
        except Exception as e:
            self.logger.error(f"Error estimating gas price: {e}")
            return None 