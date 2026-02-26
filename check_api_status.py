"""
API Status Checker
This script checks if the APIs configured in strategy/config_strategy_api.py are working properly.
"""

import sys
import os

# Add the strategy directory to the Python path to import the config
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategy'))

try:
    # Import the session from the configuration
    from config_strategy_api import session, mode, api_key, api_secret, api_url
    
    def check_api_connection():
        """
        Check if the API connection is working by making a simple request
        """
        try:
            print(f"Mode: {mode}")
            print(f"API URL: {api_url}")
            print(f"API Key configured: {'Yes' if api_key else 'No'}")
            print(f"API Secret configured: {'Yes' if api_secret else 'No'}")
            
            # Test the connection by fetching server time or getting account info
            # Using a public endpoint to test connectivity
            result = session.get_server_time()
            
            if result and 'retCode' in result and result['retCode'] == 0:
                print("✅ API Connection: SUCCESS")
                print(f"Server time: {result.get('result', {}).get('serverTime', 'N/A')}")
                return True
            else:
                print("❌ API Connection: FAILED")
                print(f"Error code: {result.get('retCode', 'Unknown')}")
                print(f"Error message: {result.get('retMsg', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"❌ API Connection: FAILED with exception - {str(e)}")
            return False
    
    def check_account_info():
        """
        Check account information if possible
        """
        try:
            # Try to get account info (this might fail if no valid API keys are provided)
            result = session.get_wallet_balance(accountType="UNIFIED")
            
            if result and 'retCode' in result and result['retCode'] == 0:
                print("\n✅ Account Info Access: SUCCESS")
                balance_result = result.get('result', {})
                if 'list' in balance_result:
                    for balance in balance_result['list']:
                        coin_balances = balance.get('coin', [])
                        if coin_balances:
                            for coin in coin_balances:
                                if float(coin.get('walletBalance', 0)) > 0:
                                    print(f"  - {coin['coin']}: {coin['walletBalance']}")
                return True
            else:
                print(f"\n⚠️  Account Info Access: FAILED - {result.get('retMsg', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"\n⚠️  Account Info Access: FAILED with exception - {str(e)}")
            return False

    def main():
        print("=" * 50)
        print("API Status Checker")
        print("=" * 50)
        
        # Check basic API connectivity
        api_connected = check_api_connection()
        
        # If basic connectivity works, try to access account info
        if api_connected:
            check_account_info()
        
        print("\n" + "=" * 50)
        print("API Status Check Complete")
        print("=" * 50)

except ImportError as e:
    print(f"❌ Failed to import config_strategy_api: {e}")
    print("Make sure the strategy/config_strategy_api.py file exists and is properly formatted.")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    try:
        main()
    except NameError:
        # This handles the case where main is not defined due to import errors
        pass