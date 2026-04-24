"""
Example: Using Tool Registry
Shows how to register custom tool implementations using the ToolRegistry.
This is cleaner than subclassing LiveInteractionSession.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import LiveInteractionSession
from core.config_manager import ConfigManager
from tools.tool_registry import ToolRegistry


# Method 1: Register tools using decorator syntax
registry = ToolRegistry()


@registry.tool("getBookingDetails")
async def get_booking_details(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Get booking details from booking system API."""
    booking_id = tool_input.get('booking_id')

    print(f"      🔍 Looking up booking: {booking_id}")

    # Simulate API call
    await asyncio.sleep(0.5)

    # Return realistic booking data
    return {
        "status": "success",
        "booking_id": booking_id,
        "hotel_name": "Grand Plaza Hotel",
        "check_in": "2025-03-15",
        "check_out": "2025-03-18",
        "guest_name": "John Doe",
        "room_type": "Deluxe Suite",
        "total_cost": 450.00,
        "currency": "USD",
        "confirmation_email": "john.doe@example.com"
    }


@registry.tool("searchDestinations")
async def search_destinations(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Search for travel destinations."""
    query = tool_input.get('query', '')

    print(f"      🔍 Searching destinations for: {query}")

    # Simulate API call
    await asyncio.sleep(0.5)

    # Return destination results
    return {
        "status": "success",
        "query": query,
        "results": [
            {
                "destination": "Paris, France",
                "description": "City of lights with iconic landmarks",
                "avg_temperature": "18°C",
                "best_time": "April-October",
                "attractions": ["Eiffel Tower", "Louvre Museum", "Notre Dame"],
                "avg_hotel_price": "$150/night"
            },
            {
                "destination": "Tokyo, Japan",
                "description": "Modern metropolis with rich culture",
                "avg_temperature": "16°C",
                "best_time": "March-May",
                "attractions": ["Tokyo Tower", "Shibuya Crossing", "Senso-ji Temple"],
                "avg_hotel_price": "$120/night"
            },
            {
                "destination": "Barcelona, Spain",
                "description": "Mediterranean coastal city with stunning architecture",
                "avg_temperature": "21°C",
                "best_time": "May-September",
                "attractions": ["Sagrada Familia", "Park Güell", "Gothic Quarter"],
                "avg_hotel_price": "$100/night"
            }
        ]
    }


@registry.tool("getRewardsBalance")
async def get_rewards_balance(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Get rewards points balance and tier information."""
    account_id = tool_input.get('account_id', 'default')

    print(f"      🔍 Fetching rewards for account: {account_id}")

    # Simulate API call
    await asyncio.sleep(0.5)

    # Return rewards data
    return {
        "status": "success",
        "account_id": account_id,
        "total_points": 125000,
        "tier": "Platinum",
        "tier_benefits": [
            "Free checked bags",
            "Priority boarding",
            "Lounge access",
            "50% bonus points on travel"
        ],
        "points_expiring_soon": 5000,
        "expiration_date": "2025-12-31",
        "redemption_value_usd": 1250.00,
        "points_needed_for_next_tier": 25000,
        "next_tier": "Diamond"
    }


# Method 2: Register tools using function calls
def register_additional_tools(registry: ToolRegistry):
    """Register additional tools programmatically."""

    async def cancel_booking(tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a booking."""
        booking_id = tool_input.get('booking_id')

        print(f"      ⚠️  Canceling booking: {booking_id}")
        await asyncio.sleep(0.5)

        return {
            "status": "success",
            "booking_id": booking_id,
            "cancellation_confirmed": True,
            "refund_amount": 450.00,
            "refund_method": "Original payment method",
            "refund_timeline": "5-7 business days"
        }

    registry.register("cancelBooking", cancel_booking)


async def main():
    """Run the session with registered tools."""

    # Register additional tools if needed
    # register_additional_tools(registry)

    print("="*70)
    print("Tool Registry Example")
    print("="*70)
    print(f"\n📋 Registered tools: {', '.join(registry.list_tools())}\n")

    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/example_with_tools.json")

    # Create session with tool registry
    session = LiveInteractionSession(config, tool_registry=registry)

    # Run the session
    await session.run()


if __name__ == "__main__":
    asyncio.run(main())
