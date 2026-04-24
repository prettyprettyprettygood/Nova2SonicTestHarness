"""
Example: Custom Tool Implementation
Shows how to implement real tool APIs instead of mocks.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import LiveInteractionSession
from core.config_manager import ConfigManager


class CustomToolSession(LiveInteractionSession):
    """
    Extended session with custom tool implementations.
    """

    async def handle_tool_call(
        self,
        tool_name: str,
        tool_content: Dict[str, Any],
        tool_use_id: str
    ) -> Dict[str, Any]:
        """
        Execute tools with real API implementations.

        Args:
            tool_name: Name of the tool
            tool_content: Tool content/parameters from Nova Sonic
            tool_use_id: Tool use ID

        Returns:
            Tool execution result
        """
        print(f"⚙️  Executing {tool_name} with real API...")

        try:
            if tool_name == "getBookingDetails":
                tool_input = tool_content.get('content', {})
                return await self._get_booking_details(tool_input)

            elif tool_name == "searchDestinations":
                tool_input = tool_content.get('content', {})
                return await self._search_destinations(tool_input)

            elif tool_name == "getRewardsBalance":
                tool_input = tool_content.get('content', {})
                return await self._get_rewards_balance(tool_input)

            else:
                # Fallback to parent implementation (mock or tool_registry)
                return await super().handle_tool_call(tool_name, tool_content, tool_use_id)

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _get_booking_details(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get booking details from booking system API.

        Args:
            tool_input: Contains booking_id

        Returns:
            Booking details
        """
        booking_id = tool_input.get('booking_id')

        # Example: Call actual booking API
        # response = await booking_api_client.get_booking(booking_id)

        # For demo purposes, return mock data
        await asyncio.sleep(0.5)  # Simulate API call

        return {
            "status": "success",
            "booking_id": booking_id,
            "hotel_name": "Grand Hotel",
            "check_in": "2025-03-15",
            "check_out": "2025-03-18",
            "guest_name": "John Doe",
            "room_type": "Deluxe Suite",
            "total_cost": 450.00,
            "currency": "USD"
        }

    async def _search_destinations(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for travel destinations.

        Args:
            tool_input: Contains query

        Returns:
            Destination search results
        """
        query = tool_input.get('query')

        # Example: Call actual search API
        # response = await travel_search_api.search(query)

        # For demo purposes, return mock data
        await asyncio.sleep(0.5)  # Simulate API call

        return {
            "status": "success",
            "query": query,
            "results": [
                {
                    "destination": "Paris, France",
                    "description": "City of lights with iconic landmarks",
                    "avg_temperature": "18°C",
                    "best_time": "April-October",
                    "attractions": ["Eiffel Tower", "Louvre Museum", "Notre Dame"]
                },
                {
                    "destination": "Tokyo, Japan",
                    "description": "Modern metropolis with rich culture",
                    "avg_temperature": "16°C",
                    "best_time": "March-May",
                    "attractions": ["Tokyo Tower", "Shibuya", "Senso-ji Temple"]
                }
            ]
        }

    async def _get_rewards_balance(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get rewards points balance.

        Args:
            tool_input: Optional account parameters

        Returns:
            Rewards balance information
        """
        # Example: Call actual rewards API
        # response = await rewards_api.get_balance(account_id)

        # For demo purposes, return mock data
        await asyncio.sleep(0.5)  # Simulate API call

        return {
            "status": "success",
            "total_points": 125000,
            "tier": "Platinum",
            "points_expiring_soon": 5000,
            "expiration_date": "2025-12-31",
            "redemption_value_usd": 1250.00
        }


async def main_with_custom_tools():
    """
    Example: Run session with custom tool implementations.
    """
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/example_with_tools.json")

    # Create custom session
    session = CustomToolSession(config)

    # Run the session
    await session.run()


if __name__ == "__main__":
    asyncio.run(main_with_custom_tools())
