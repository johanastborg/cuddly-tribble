from asql.parser import parse_query
from asql.engine import ASQLEngine
from asql.data import generate_mock_data
import json

def main():
    query = """FROM "SuperpositionEngine" AS engine
| RANGE last 5m
| WINDOW every 30s
| MIN() AS "noise_floor"
| MAX() AS "peak_amplitude"
| EMIT "GlitchMonitor" """

    print("--- ASQL Query ---")
    print(query)
    
    # 1. Parse the query
    print("\n--- Parsing ---")
    try:
        plan = parse_query(query)
        # Convert to a readable string for display (Lark objects aren't directly serializable easily)
        print("Plan parsed successfully.")
    except Exception as e:
        print(f"Parsing error: {e}")
        return

    # 2. Setup mock data and engine
    print("\n--- data generation ---")
    mock_data = generate_mock_data()
    engine = ASQLEngine(mock_data)

    # 3. Execute
    print("\n--- Execution ---")
    result_state = engine.execute(plan)

    # 4. Results
    print("\n--- Result State ---")
    print(f"Active status: {result_state['active']}")
    print("Done.")

if __name__ == "__main__":
    main()
