import json
import logging
from pathlib import Path
from datetime import datetime
from autotrade.analysis.post_market import PostMarketAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PostMarketEval")

def run_evaluation():
    # Use today's date by default or provided date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Paths for standard PMR
    plan_path = Path(f"plans/pm_plan_{today}.json")
    review_path = Path(f"data/eod_review_{today}.json")
    
    # Path for Overnight Watchlist
    overnight_path = Path("research/overnight_state.json")

    analyzer = PostMarketAnalyzer()
    
    overnight_metrics = {}
    if overnight_path.exists():
        with open(overnight_path, "r") as f:
            overnight_data = json.load(f)
            watchlist = overnight_data.get("watchlist", [])
            # The file might have a different date inside, but we assume it's for the current session
            overnight_metrics = analyzer.evaluate_overnight_watchlist(watchlist, today)

    # Standard PMR Logic (if files exist)
    standard_pmr_active = plan_path.exists() and review_path.exists()
    adherence = {}
    metrics = {}
    lesson = ""

    if standard_pmr_active:
        with open(plan_path, "r") as f:
            plan = json.load(f)
        
        with open(review_path, "r") as f:
            review_data = json.load(f)
            executions = review_data.get("trades", [])

        adherence = analyzer.calculate_plan_adherence(plan, executions)
        metrics = analyzer.attribute_performance(plan, executions)
        lesson = analyzer.synthesize_lesson(metrics)

    # Output for User
    print("\n" + "="*60)
    print(f"POST-MARKET EVALUATION: {today}")
    print("="*60)
    
    if overnight_metrics:
        print(f"\nTOP 50 OVERNIGHT WATCHLIST PERFORMANCE:")
        print(f"  Avg Peak (OH): {overnight_metrics['avg_peak_return']*100:.2f}%")
        print(f"  Avg Close (OC): {overnight_metrics['avg_close_return']*100:.2f}%")
        print(f"  Win Rate (OC > 0): {overnight_metrics['win_rate']*100:.2f}%")
        print(f"  Hit Rate (OH >= 2%): {overnight_metrics['hit_rate_2pct']*100:.2f}%")
        
        print("\n  Top Performers (Peak):")
        for p in overnight_metrics['top_performers']:
            print(f"    {p['symbol']:<5} | OH: {p['perf_oh']*100:>6.2f}%")

    if standard_pmr_active:
        print("\nINTRADAY PLAN ADHERENCE:")
        print(f"  Adherence Score: {adherence['adherence_score']:.2f}")
        print(f"  Performance Delta: {metrics['performance_delta']*100:.2f}%")
        print("\nSYSTEM LESSON:")
        print(f"  {lesson}")
    else:
        print("\nINTRADAY EXECUTION DATA MISSING - Skipping Adherence Review.")

    print("\n" + "="*60)

    # Save consolidated report
    report_data = {
        "date": today,
        "overnight_watchlist": overnight_metrics,
        "execution_adherence": adherence,
        "execution_metrics": metrics,
        "system_lesson": lesson
    }
    report_path = Path(f"reports/post_market_summary_{today.replace('-', '')}.json")
    with report_path.open("w") as f:
        json.dump(report_data, f, indent=4)
    
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_evaluation()
