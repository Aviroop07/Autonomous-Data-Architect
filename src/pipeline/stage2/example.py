from src.pipeline.prompt_engineer.models import PromptExample
from .models.schema import SchemaSegment, Table, Column, ForeignKey

# Example 1: Dimension Tables
# Refined to be more descriptive and strict about entity discovery.
dimension_example = PromptExample[SchemaSegment](
    scenario=(
        "The system requires a way to track basic equity information and their listing exchanges. "
        "Specifically, we need an 'EQUITY' table to store tickers and a 'MARKET_EXCHANGE' table for exchange names. "
        "Each equity must be linked to exactly one exchange for its primary listing."
    ),
    instance=SchemaSegment(
        chunk_title="Core Dimensions",
        tables=[
            Table(
                name="EQUITY",
                columns=[Column(name="equity_id"), Column(name="ticker"), Column(name="exchange_id")],
                pk="equity_id"
            ),
            Table(
                name="MARKET_EXCHANGE",
                columns=[Column(name="exchange_id"), Column(name="name")],
                pk="exchange_id"
            )
        ],
        relationships=[
            ForeignKey(referencing_table="EQUITY", referencing_column="exchange_id", referred_table="MARKET_EXCHANGE")
        ]
    ),
    reasoning="The example strictly models the 'EQUITY' and 'MARKET_EXCHANGE' entities mentioned in the scenario. It uses standard PK/FK patterns and doesn't add unmentioned columns or tables."
)

# Example 2: Fact Tables
# Refined for high-fidelity description and strict entity mapping.
fact_example = PromptExample[SchemaSegment](
    scenario=(
        "We need to capture high-frequency trade executions in a 'TRADE_FACT' table. "
        "This table must record the unique trade ID, the price, the quantity, and a high-precision timestamp. "
        "It also needs to reference the 'EQUITY' from the dimension chunk."
    ),
    instance=SchemaSegment(
        chunk_title="Market Trades",
        tables=[
            Table(
                name="TRADE_FACT",
                columns=[
                    Column(name="trade_id"), 
                    Column(name="equity_id"), 
                    Column(name="timestamp_ns"), 
                    Column(name="price"), 
                    Column(name="quantity")
                ],
                pk="trade_id"
            )
        ],
        relationships=[
            # Note: Referred table 'EQUITY' is external to this segment but mentioned in scenario
            ForeignKey(referencing_table="TRADE_FACT", referencing_column="equity_id", referred_table="EQUITY")
        ]
    ),
    reasoning="This chunk translates the 'TRADE_FACT' requirement into a single-table schema segment. It only includes columns explicitly required (price, quantity, timestamp, etc.) and links to the 'EQUITY' entity as requested."
)

# Export as a list for the Prompt Engineer
STAGE2_EXAMPLES = [dimension_example, fact_example]
