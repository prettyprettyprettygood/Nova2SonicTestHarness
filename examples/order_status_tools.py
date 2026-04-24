"""
Order Status Tools for Judge Calibration

Tool handlers for the order status inquiry customer service scenario.
Provides a lookupOrder tool with mock order data across different statuses
(shipped, processing, delivered, delayed).
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.tool_registry import ToolRegistry


# Module-level registry (loaded by main.py via tool_registry_module config)
registry = ToolRegistry()

# Mock order database with varied statuses for diverse test scenarios
MOCK_ORDERS = {
    "ORD-2024-78432": {
        "status": "success",
        "order_id": "ORD-2024-78432",
        "order_status": "shipped",
        "customer_name": "Sarah Johnson",
        "order_date": "2026-02-20",
        "items": [
            {"name": "Wireless Bluetooth Headphones", "quantity": 1, "price": 79.99},
            {"name": "Phone Case - Clear", "quantity": 2, "price": 12.99}
        ],
        "shipping": {
            "carrier": "FedEx",
            "tracking_number": "FX-9876543210",
            "shipped_date": "2026-02-22",
            "estimated_delivery": "2026-02-27",
            "current_location": "Memphis, TN distribution center"
        },
        "order_total": 105.97,
        "payment_method": "Visa ending in 4532"
    },
    "ORD-2024-91205": {
        "status": "success",
        "order_id": "ORD-2024-91205",
        "order_status": "processing",
        "customer_name": "Mike Chen",
        "order_date": "2026-02-24",
        "items": [
            {"name": "USB-C Charging Cable 6ft", "quantity": 3, "price": 8.99}
        ],
        "shipping": {
            "carrier": "pending",
            "tracking_number": None,
            "shipped_date": None,
            "estimated_delivery": "2026-03-01",
            "current_location": None
        },
        "order_total": 26.97,
        "payment_method": "Mastercard ending in 8821"
    },
    "ORD-2024-55678": {
        "status": "success",
        "order_id": "ORD-2024-55678",
        "order_status": "delivered",
        "customer_name": "Emily Davis",
        "order_date": "2026-02-15",
        "items": [
            {"name": "Yoga Mat - Purple", "quantity": 1, "price": 34.99},
            {"name": "Water Bottle 32oz", "quantity": 1, "price": 19.99}
        ],
        "shipping": {
            "carrier": "UPS",
            "tracking_number": "UPS-1234567890",
            "shipped_date": "2026-02-17",
            "estimated_delivery": "2026-02-21",
            "delivered_date": "2026-02-20",
            "current_location": "Delivered - Front porch"
        },
        "order_total": 54.98,
        "payment_method": "PayPal"
    },
    "ORD-2024-33100": {
        "status": "success",
        "order_id": "ORD-2024-33100",
        "order_status": "delayed",
        "customer_name": "James Wilson",
        "order_date": "2026-02-10",
        "items": [
            {"name": "Standing Desk Converter", "quantity": 1, "price": 249.99}
        ],
        "shipping": {
            "carrier": "FedEx",
            "tracking_number": "FX-1122334455",
            "shipped_date": "2026-02-12",
            "estimated_delivery": "2026-02-18",
            "revised_delivery": "2026-02-28",
            "delay_reason": "Weather delay at regional hub",
            "current_location": "Chicago, IL - held due to weather"
        },
        "order_total": 249.99,
        "payment_method": "Visa ending in 7890"
    }
}


@registry.tool("lookupOrder")
async def lookup_order(tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Look up order details and status by order ID."""
    order_id = tool_input.get("order_id", "")

    print(f"      Looking up order: {order_id}")

    # Simulate API latency
    await asyncio.sleep(0.5)

    if order_id in MOCK_ORDERS:
        return MOCK_ORDERS[order_id]

    return {
        "status": "error",
        "error": "Order not found",
        "order_id": order_id,
        "message": "No order found with this ID. Please verify the order number."
    }


async def main():
    """Standalone test for the tool handler."""
    print("Order Status Tools")
    print(f"Registered tools: {', '.join(registry.list_tools())}")
    print()

    # Test each order
    for order_id in ["ORD-2024-78432", "ORD-2024-91205", "ORD-2024-55678", "ORD-2024-33100", "ORD-UNKNOWN"]:
        result = await registry.execute("lookupOrder", {"order_id": order_id})
        status = result.get("order_status", result.get("error", "unknown"))
        print(f"  {order_id} -> {status}")


if __name__ == "__main__":
    asyncio.run(main())
