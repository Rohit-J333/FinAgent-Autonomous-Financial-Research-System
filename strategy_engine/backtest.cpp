/*
 * FinAgent C++ Strategy Backtester
 *
 * Reads a JSON input file, runs the specified strategy simulation,
 * and writes results to a JSON output file.
 *
 * Usage:
 *   ./backtest <input.json> <output.json>
 *
 * Supported strategies:
 *   - momentum       : Buy when price > N-day SMA, sell when below
 *   - mean_reversion : Buy when price < SMA - k*std, sell when > SMA + k*std
 *   - macd           : Buy/sell on MACD crossover signals
 *
 * Build:
 *   mkdir -p build && cd build && cmake .. && make
 */

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no external dependency)
// ---------------------------------------------------------------------------

string readFile(const string& path) {
    ifstream f(path);
    if (!f.is_open()) {
        cerr << "Cannot open file: " << path << "\n";
        return "";
    }
    ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

string extractString(const string& json, const string& key) {
    string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == string::npos) return "";
    pos = json.find(":", pos);
    if (pos == string::npos) return "";
    pos = json.find("\"", pos);
    if (pos == string::npos) return "";
    auto end = json.find("\"", pos + 1);
    return json.substr(pos + 1, end - pos - 1);
}

int extractInt(const string& json, const string& key, int def = 20) {
    string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == string::npos) return def;
    pos = json.find(":", pos);
    if (pos == string::npos) return def;
    // skip whitespace
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    return stoi(json.substr(pos));
}

// ---------------------------------------------------------------------------
// Price simulation (deterministic, seeded by symbol hash)
// ---------------------------------------------------------------------------

vector<double> generatePrices(const string& symbol, int days) {
    // Seed with symbol hash for determinism
    size_t seed = hash<string>{}(symbol);
    mt19937 rng(seed);
    normal_distribution<double> dist(0.0, 0.015); // ~1.5% daily vol

    vector<double> prices;
    prices.reserve(days);
    double price = 150.0;
    for (int i = 0; i < days; ++i) {
        price *= (1.0 + dist(rng));
        prices.push_back(price);
    }
    return prices;
}

// ---------------------------------------------------------------------------
// Strategy implementations
// ---------------------------------------------------------------------------

struct Trade {
    int entry_day;
    int exit_day;
    double entry_price;
    double exit_price;
    bool is_long;
};

double sma(const vector<double>& prices, int end, int period) {
    if (end < period) return prices[end];
    double sum = 0;
    for (int i = end - period; i < end; ++i) sum += prices[i];
    return sum / period;
}

double stddev(const vector<double>& prices, int end, int period) {
    double mean = sma(prices, end, period);
    double sq_sum = 0;
    for (int i = end - period; i < end; ++i)
        sq_sum += (prices[i] - mean) * (prices[i] - mean);
    return sqrt(sq_sum / period);
}

vector<Trade> runMomentum(const vector<double>& prices, int period) {
    vector<Trade> trades;
    bool in_trade = false;
    int entry_day = 0;
    double entry_price = 0;

    for (int i = period; i < (int)prices.size() - 1; ++i) {
        double avg = sma(prices, i, period);
        if (!in_trade && prices[i] > avg) {
            in_trade = true;
            entry_day = i;
            entry_price = prices[i];
        } else if (in_trade && prices[i] < avg) {
            trades.push_back({entry_day, i, entry_price, prices[i], true});
            in_trade = false;
        }
    }
    if (in_trade)
        trades.push_back({entry_day, (int)prices.size()-1, entry_price, prices.back(), true});
    return trades;
}

vector<Trade> runMeanReversion(const vector<double>& prices, int period) {
    vector<Trade> trades;
    bool in_trade = false;
    int entry_day = 0;
    double entry_price = 0;
    double k = 1.5;

    for (int i = period; i < (int)prices.size() - 1; ++i) {
        double avg = sma(prices, i, period);
        double sd  = stddev(prices, i, period);
        if (!in_trade && prices[i] < avg - k * sd) {
            in_trade = true;
            entry_day = i;
            entry_price = prices[i];
        } else if (in_trade && prices[i] > avg) {
            trades.push_back({entry_day, i, entry_price, prices[i], true});
            in_trade = false;
        }
    }
    if (in_trade)
        trades.push_back({entry_day, (int)prices.size()-1, entry_price, prices.back(), true});
    return trades;
}

vector<Trade> runMACD(const vector<double>& prices) {
    // EMA helper
    auto ema = [&](int period, int end) {
        double k = 2.0 / (period + 1);
        double e = prices[0];
        for (int i = 1; i <= end; ++i)
            e = prices[i] * k + e * (1 - k);
        return e;
    };

    vector<Trade> trades;
    bool in_trade = false;
    int entry_day = 0;
    double entry_price = 0;

    for (int i = 26; i < (int)prices.size() - 1; ++i) {
        double macd   = ema(12, i) - ema(26, i);
        double signal = ema(9, i); // simplified
        if (!in_trade && macd > signal) {
            in_trade = true;
            entry_day = i;
            entry_price = prices[i];
        } else if (in_trade && macd < signal) {
            trades.push_back({entry_day, i, entry_price, prices[i], true});
            in_trade = false;
        }
    }
    if (in_trade)
        trades.push_back({entry_day, (int)prices.size()-1, entry_price, prices.back(), true});
    return trades;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

struct Metrics {
    double total_return;
    double sharpe_ratio;
    double max_drawdown;
    double win_rate;
    int    total_trades;
};

Metrics computeMetrics(const vector<Trade>& trades) {
    if (trades.empty())
        return {0, 0, 0, 0, 0};

    vector<double> returns;
    double cumulative = 1.0;
    double peak = 1.0;
    double max_dd = 0.0;
    int wins = 0;

    for (auto& t : trades) {
        double ret = (t.exit_price - t.entry_price) / t.entry_price;
        returns.push_back(ret);
        cumulative *= (1 + ret);
        if (cumulative > peak) peak = cumulative;
        double dd = (peak - cumulative) / peak;
        if (dd > max_dd) max_dd = dd;
        if (ret > 0) wins++;
    }

    double mean_ret = accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double sq_sum = 0;
    for (double r : returns) sq_sum += (r - mean_ret) * (r - mean_ret);
    double vol = sqrt(sq_sum / returns.size());
    double sharpe = (vol > 0) ? (mean_ret / vol) * sqrt(252.0) : 0;

    return {
        cumulative - 1.0,
        sharpe,
        -max_dd,
        (double)wins / trades.size(),
        (int)trades.size()
    };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: backtest <input.json> <output.json>\n";
        return 1;
    }

    string input_json = readFile(argv[1]);
    if (input_json.empty()) return 1;

    string strategy = extractString(input_json, "strategy");
    string symbol   = extractString(input_json, "symbol");
    int lookback    = extractInt(input_json, "lookback_period", 20);

    if (strategy.empty()) strategy = "momentum";
    if (symbol.empty())   symbol   = "AAPL";

    // Generate 1 year of synthetic prices (deterministic)
    auto prices = generatePrices(symbol, 252);

    vector<Trade> trades;
    if (strategy == "momentum")
        trades = runMomentum(prices, lookback);
    else if (strategy == "mean_reversion")
        trades = runMeanReversion(prices, lookback);
    else
        trades = runMACD(prices);

    Metrics m = computeMetrics(trades);

    // Write JSON output
    ofstream out(argv[2]);
    out << "{\n";
    out << "  \"strategy\": \"" << strategy << "\",\n";
    out << "  \"symbol\": \""   << symbol   << "\",\n";
    out << "  \"total_return\": "  << m.total_return  << ",\n";
    out << "  \"sharpe_ratio\": "  << m.sharpe_ratio  << ",\n";
    out << "  \"max_drawdown\": "  << m.max_drawdown  << ",\n";
    out << "  \"win_rate\": "      << m.win_rate      << ",\n";
    out << "  \"total_trades\": "  << m.total_trades  << "\n";
    out << "}\n";

    cout << "Backtest complete: " << m.total_trades << " trades, "
         << "return=" << m.total_return << ", sharpe=" << m.sharpe_ratio << "\n";

    return 0;
}
