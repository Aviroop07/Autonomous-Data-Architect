"""
Human-style stress tests for Stages 1 + 2.
Three cases with distinct writing styles and domain complexity.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Case 1: Ride-sharing / mobility platform — casual, PM-ish, some vagueness
# ---------------------------------------------------------------------------

NL_RIDESHARE = """
ok so basically we're building a ride-hailing platform, think uber but we also
do scheduled rides, carpooling, and corporate accounts. here's a rough dump of
what we need to track.

Drivers register with their full name, email, phone, date of birth, government-issued
ID number, and profile photo. They submit their vehicle info — make, model, year, color,
license plate, and VIN. We check their driving license (number, state, expiry), insurance
policy (number, provider, expiry), and background check result (pass/fail, date, provider).
Each driver has a rating (running average), total trips completed, total earnings, account
status (active, suspended, banned), and payout bank account details.

Riders have a profile with name, email, phone, preferred payment method, home address,
work address. They also have a running rating. Riders can belong to a corporate account
— so a company registers with a billing address, tax ID, and a spending limit per
employee per month.

Trips are the main thing. A trip has a pickup location (lat/lng + human readable address),
dropoff location (same), requested time, actual start time, actual end time, distance in km,
base fare, surge multiplier (1.0 if no surge), tolls, tips, total amount charged, the
payment transaction ID, status (requested, accepted, in_progress, completed, cancelled,
no_show). A trip is always between one rider and one driver. For carpools, multiple riders
can share a trip — each rider has their own pickup/dropoff within the trip and their own
share of the fare.

We need to track driver location pings — every 30 seconds when the driver app is open, we
record lat/lng and timestamp. Useful for ETA calculation and auditing disputed trips.

For surge pricing we track zones — a zone is a geographic polygon (stored as a WKT string
I guess), has a name, city, country. At any given time a zone can have a surge multiplier
set. We keep a history of when surge was active in each zone.

Trip requests — before a trip is confirmed, a rider sends a request. The request has the
pickup, dropoff, requested vehicle class (economy, comfort, xl, premium), estimated fare,
timestamp. The system then assigns a driver. We want to log which drivers were offered the
trip and whether they accepted, declined, or didn't respond (timeout).

Payments — when a trip ends we charge the rider's payment method. Payment methods can be
credit card (tokenized), debit card, digital wallet (PayPal, Apple Pay, Google Pay),
or corporate account. We need to track each charge attempt — amount, status (success, failed,
refunded), processor reference, timestamp. Failed charges trigger a retry up to 3 times.

Rider and driver support tickets — both can open a dispute. A ticket has a category (wrong
route, safety concern, lost item, billing issue, driver behavior, rider behavior, app bug),
description, status (open, in_review, resolved, escalated), priority, assigned support agent,
resolution notes, and timestamps. A ticket can be linked to a trip. Escalated tickets get
a supervisor.

Promotions and promo codes — a promo has a code, discount type (flat or percent), discount
value, max uses total, max uses per rider, expiry date, and which cities it applies to.
We track each time a promo is redeemed (who, when, for which trip, amount discounted).

Driver earnings and payouts. After each trip the driver's earnings are updated. Payouts
happen weekly — a payout record has the driver, period start/end, gross earnings, platform
fee (%), net amount, payout status (pending, processed, failed), and transaction reference.

Ratings — after each completed trip, both rider and driver can rate each other. Rating
has a score (1-5), optional text comment, timestamp, and flags (e.g. rude, great conversation,
clean vehicle). If a rating contains a safety flag, it auto-creates a support ticket.

We also need a vehicle inspection log. Periodically drivers submit photos and an
inspection form. Each inspection has date, inspector (could be automated or a person),
result (pass/fail), list of items checked, and any issues flagged.
"""

# ---------------------------------------------------------------------------
# Case 2: Investment portfolio management — domain-expert writing, dense
# ---------------------------------------------------------------------------

NL_PORTFOLIO = """
Multi-asset portfolio management system for a mid-size wealth management firm.
We run discretionary and advisory mandates across equities, fixed income, derivatives,
FX, and alternative investments for circa 800 HNW and UHNW clients.

Client onboarding captures legal name, date of birth, nationality, tax residency (can
be multiple jurisdictions), TIN per jurisdiction, risk appetite (1-7 scale per ESMA
guidelines), investment horizon, liquidity requirements, ESG preference flag, restricted
securities list (companies the client cannot hold due to employment or personal preference),
KYC status (pending, verified, expired), AML check result and date, PEP flag, and
relationship manager assignment.

Accounts. A client can have multiple accounts — each account is denominated in a base
currency, has an account type (discretionary, advisory, custody), an IPS (investment
policy statement) document reference, benchmark index, management fee tier (bps), and
opening/closing dates.

Securities master. We need a securities reference table: ISIN, CUSIP, SEDOL, ticker,
name, asset class (equity, bond, ETF, future, option, FX spot, FX forward, structured note,
hedge fund unit, private equity), issuer, currency, exchange, country of risk, GICS sector
and industry, duration for bonds, coupon rate, maturity date, option style (European/American),
underlying security for derivatives, contract multiplier, and whether the security is on our
approved list.

Positions. At end of each business day we snapshot each account's holdings: security,
quantity, average cost, current market price (with source — bloomberg, reuters, manual),
market value, unrealized P&L, realized P&L YTD, weight in portfolio. FX positions are
recorded as notional amounts in both base and foreign currency.

Transactions. Every buy, sell, subscription, redemption, dividend receipt, coupon payment,
FX conversion, fee deduction, and corporate action adjustment is a transaction record.
Fields: account, security, transaction type, trade date, settlement date, quantity,
price, gross amount, fx rate applied (if cross-currency), net amount after fees/taxes,
accrued interest (for bonds), broker/counterparty, order reference, execution venue,
settlement status (pending, settled, failed), and the portfolio manager who authorized.

Orders. Before execution we record an order: account, security, direction (buy/sell),
order type (market, limit, stop), quantity or target weight, limit price if applicable,
order status (draft, submitted, partially_filled, filled, cancelled, expired), time in force
(GTC, GTD, day), creation timestamp, and the strategy or rebalance trigger that generated it.
Orders can be split into multiple execution fills.

Benchmarks and performance. For each account we calculate daily, MTD, QTD, YTD, and
since-inception returns (both absolute and relative to benchmark). We store the NAV per
unit/share for pooled vehicles. Attribution analysis at the sector and security level:
allocation effect, selection effect, interaction effect.

Risk metrics. Daily VaR (parametric 95% and 99%, historical simulation, Monte Carlo),
CVaR, beta vs benchmark, tracking error, information ratio, Sharpe ratio, Sortino ratio,
max drawdown and drawdown duration. Also stress test results — we run standard shock
scenarios (rates +/-100/200bps, equity -20/-30%, FX +/-10%) and record the simulated
portfolio impact per scenario per account.

Corporate actions. Stock splits, mergers, spin-offs, dividends (cash and stock), rights
issues, tender offers. Each corporate action has an ex-date, record date, pay date, ratio
or amount, affected security, and the accounts impacted. Resulting position adjustments
are linked back to the corporate action record.

Compliance and limits. Each account has investment guidelines encoded as limits:
max weight per single security, per sector, per country, per asset class; min credit
rating for bonds; prohibited securities and sectors; leverage limit. Daily compliance
checks compare current positions against limits and log breaches with severity (warning,
breach) and resolution status.

Fee calculations. Management fee is accrued daily on AUM at the applicable tier.
Performance fee is calculated annually (or quarterly for some mandates) using a
high-watermark structure. Each accrual and crystallization event is recorded with
the fee type, period, AUM or performance base, rate, gross amount, tax withheld, and
net amount invoiced.

Counterparty registry. Brokers, custodians, prime brokers, fund administrators — each
with LEI, legal name, country, relationship type, credit limit, settlement instructions
(BIC, IBAN per currency), and our credit exposure to them.
"""

# ---------------------------------------------------------------------------
# Case 3: Multi-tenant B2B SaaS project management — informal PM notes
# ---------------------------------------------------------------------------

NL_SAAS_PM = """
Building a project management SaaS, think Jira/Linear but simpler. Multi-tenant,
each paying customer is a "workspace". Here's everything the data model needs to support.

Workspaces — each has a name, slug (for their subdomain), plan tier (free, starter, pro,
enterprise), billing email, created date, owner user, max seats allowed by plan, and
a settings blob (JSON-ish — theme color, notification defaults, feature flags). Workspaces
on the enterprise plan get a custom domain.

Users exist globally but belong to one or more workspaces with a role. User profile:
email (unique globally), full name, avatar URL, timezone, notification preferences,
last active timestamp, account status (active, deactivated). Within a workspace, a
user has a role — admin, member, viewer, or guest (guests can only see projects explicitly
shared with them). Users can be deactivated globally (e.g. after account deletion request).

Projects live inside a workspace. A project has a name, description, key (short uppercase
prefix like "ENG", "MKT"), icon, color, status (active, archived, deleted), start date,
target date, owner, and optionally a parent project (for sub-projects). Projects can be
set to private (only explicit members see them) or workspace-visible.

Issues are the core entity. An issue belongs to a project, has a title, description
(rich text, stored as markdown or a JSON document format), identifier (e.g. ENG-142),
status, priority (urgent, high, medium, low, none), assignee (a workspace member),
reporter, created date, updated date, due date, estimated effort (story points or hours,
we haven't decided which yet), and cycle time (auto-calculated when closed).

Issue statuses are customizable per project — we'll have a default set (backlog, todo,
in_progress, in_review, done, cancelled) but projects can rename or add statuses.
Each status has a type (unstarted, started, completed, cancelled) and a position for ordering.

Labels — workspace-level tags with a name and color. Issues can have multiple labels.

Issues can have parent issues (epics), and child issues (sub-tasks). An issue can also
block or be blocked by other issues.

Sprints / cycles — a project can use sprint-based planning. A sprint has a name, start date,
end date, status (planned, active, completed). Issues can be added to a sprint. When a
sprint ends, incomplete issues should be tracked (moved to next sprint or left in backlog).

Comments on issues — author, content (markdown), created/updated timestamps. Comments
can be reactions (emoji, similar to Slack). Comments can also be edited or deleted
(but we keep an audit trail of edits). Comments support @mention of users.

Attachments — files attached to an issue or comment. File name, size, MIME type,
storage URL, uploader, upload timestamp.

Notifications — when something happens (issue assigned to you, comment mentioning you,
issue status change, due date approaching), a notification is created. Fields: recipient,
type, the triggering object (polymorphic — could be an issue, comment, etc.), read status,
created timestamp, and the URL to navigate to.

Activity log / audit trail — every change to an issue should be recorded: what changed
(field name), old value, new value, who changed it, when. This powers the issue timeline
view. Changes tracked: status, assignee, priority, due date, sprint, parent, title,
labels added/removed.

Integrations — workspaces can connect external services. We support GitHub (link issues
to PRs/commits), Slack (notifications), and Figma (link designs to issues). Each
integration stores credentials (encrypted), webhook URLs, settings, and sync status.
GitHub integration specifically needs to store repo mappings — which GitHub repos
are linked to which projects.

Search — we'll need to support full-text search on issue titles and descriptions.
Plan to use Postgres full-text search initially, so we need tsvector columns.
Also want to be able to filter by any field combination and save those filters
as "views" — a saved view has a name, filter JSON, sort config, grouping,
and can be personal or shared to the workspace.

Billing — for SaaS billing we need: subscription (workspace, plan, billing cycle
monthly/annual, status, current period start/end, Stripe subscription ID), invoices
(amount, currency, status paid/unpaid/void, PDF URL, Stripe invoice ID), and
payment methods (Stripe payment method ID, type card/bank, last 4, expiry).

Teams — workspaces can have teams (sub-groups of members). A team has a name,
description, icon, and a list of members with their team role. Projects can be
assigned to teams. Issues can be assigned to teams (in addition to or instead of
individuals).
"""


CASES = [
    ("rideshare", NL_RIDESHARE),
    ("portfolio_mgmt", NL_PORTFOLIO),
    ("saas_pm", NL_SAAS_PM),
]


async def run_case(
    label: str,
    nl: str,
    model: str = "gpt-4o",
) -> dict:
    from src.orchestration.stage1.entry import orchestrate as stage1
    from src.orchestration.stage2.entry import orchestrate as stage2
    from src.util.config.ablation import AblationConfig

    ablation = AblationConfig.full()
    out_dir = PROJECT_ROOT / "output" / "runs" / f"human_stress_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  [{label}] Stage 1 starting ({len(nl)} chars)")
    print(f"{'=' * 60}")
    t0 = time.time()

    s1_output, s1_tokens = await stage1(
        nl_description=nl,
        model=model,
        ablation_config=ablation,
    )
    s1_elapsed = time.time() - t0
    facts = s1_output.final_facts
    print(
        f"[{label}] Stage 1 done: {len(facts)} facts | "
        f"{s1_elapsed:.1f}s | {s1_tokens} tokens | "
        f"domain={s1_output.domain!r}"
    )
    (out_dir / "stage1_output.json").write_text(
        json.dumps(s1_output.model_dump(), indent=2, default=str), encoding="utf-8"
    )

    print(f"[{label}] Stage 2 starting...")
    t1 = time.time()
    s2_output, s2_tokens, _registry = await stage2(
        facts=facts,
        domain=s1_output.domain,
        analytical_goal=s1_output.analytical_goal,
        model=model,
        ablation_config=ablation,
    )
    s2_elapsed = time.time() - t1
    schema = s2_output.final_global_schema
    assert schema is not None, "Stage 2 returned no schema"
    validation_errors = schema._validate()

    print(
        f"[{label}] Stage 2 done: {len(schema.tables)} tables | "
        f"{len(schema.relationships or [])} FKs | "
        f"{len(validation_errors)} errs | "
        f"{s2_elapsed:.1f}s | {s2_tokens} tokens"
    )
    (out_dir / "stage2_output.json").write_text(
        json.dumps(s2_output.model_dump(), indent=2, default=str), encoding="utf-8"
    )

    return {
        "label": label,
        "nl_chars": len(nl),
        "domain": s1_output.domain,
        "analytical_goal": s1_output.analytical_goal,
        "s1_facts": len(facts),
        "s1_tokens": s1_tokens,
        "s1_time": round(s1_elapsed, 1),
        "tables": len(schema.tables),
        "relationships": len(schema.relationships or []),
        "validation_errors": len(validation_errors),
        "error_list": validation_errors[:5],
        "s2_tokens": s2_tokens,
        "s2_time": round(s2_elapsed, 1),
        "total_tokens": s1_tokens + s2_tokens,
        "total_time": round(s1_elapsed + s2_elapsed, 1),
        "table_list": sorted(
            [
                {
                    "name": t.name,
                    "cols": len(t.columns),
                    "fks": sum(
                        1
                        for r in (schema.relationships or [])
                        if r.referencing_table == t.name
                    ),
                }
                for t in schema.tables
            ],
            key=lambda x: x["name"],
        ),
    }


async def main() -> None:
    import warnings

    warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

    results = []
    for label, nl in CASES:
        try:
            result = await run_case(label, nl)
            results.append(result)
        except Exception as exc:
            print(f"[{label}] FAILED: {exc}")
            results.append({"label": label, "error": str(exc)})

    print("\n\n" + "=" * 70)
    print("  STRESS TEST SUMMARY")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"\n  {r['label']}: ERROR — {r['error']}")
            continue
        print(f"\n  {r['label']}  ({r['nl_chars']} chars)")
        print(f"    domain          : {r['domain']}")
        print(f"    analytical_goal : {r['analytical_goal']}")
        print(f"    facts           : {r['s1_facts']}")
        print(f"    tables          : {r['tables']}")
        print(f"    relationships   : {r['relationships']}")
        print(f"    validation errs : {r['validation_errors']}")
        if r["error_list"]:
            for e in r["error_list"]:
                print(f"      - {e}")
        print(
            f"    tokens          : {r['total_tokens']} (S1={r['s1_tokens']} S2={r['s2_tokens']})"
        )
        print(
            f"    time            : {r['total_time']}s (S1={r['s1_time']}s S2={r['s2_time']}s)"
        )
        print("    tables detail:")
        for t in r["table_list"]:
            print(f"      {t['name']:<42} cols={t['cols']:<4} fks={t['fks']}")

    out_path = PROJECT_ROOT / "output" / "runs" / "human_stress_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n  Summary saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
