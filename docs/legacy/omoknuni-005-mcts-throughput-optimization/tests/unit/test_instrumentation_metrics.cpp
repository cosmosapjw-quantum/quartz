#include <gtest/gtest.h>

#include "mcts/instrumentation.hpp"

namespace {

using namespace mcts;

class InstrumentationGuard {
public:
    InstrumentationGuard() : previous_state_(Instrumentation::instance().is_enabled()) {
        Instrumentation::instance().set_enabled(true);
        Instrumentation::instance().reset();
    }

    ~InstrumentationGuard() {
        Instrumentation::instance().reset();
        Instrumentation::instance().set_enabled(previous_state_);
    }

private:
    bool previous_state_;
};

TEST(InstrumentationMetricsTest, RecordsScopedMetricDurations) {
    InstrumentationGuard guard;
    {
        ScopedMetric metric(InstrumentationMetric::Selection);
    }

    const auto snapshot = Instrumentation::instance().snapshot();
    ASSERT_FALSE(snapshot.empty());
    const auto it = snapshot.find(InstrumentationMetric::Selection);
    ASSERT_NE(it, snapshot.end());
    EXPECT_EQ(it->second.call_count, 1u);
    EXPECT_GT(it->second.total_elapsed_ns, 0u);
}

TEST(InstrumentationMetricsTest, IncrementCounterTracksCounts) {
    InstrumentationGuard guard;
    Instrumentation::instance().increment_counter(InstrumentationMetric::VirtualLossApply, 3);
    const auto snapshot = Instrumentation::instance().snapshot();
    const auto it = snapshot.find(InstrumentationMetric::VirtualLossApply);
    ASSERT_NE(it, snapshot.end());
    EXPECT_EQ(it->second.call_count, 3u);
    EXPECT_EQ(it->second.total_elapsed_ns, 0u);
}

TEST(InstrumentationMetricsTest, TracksExpansionConflicts) {
    InstrumentationGuard guard;

    // Simulate 5 expansion conflicts
    for (int i = 0; i < 5; ++i) {
        Instrumentation::instance().increment_counter(InstrumentationMetric::ExpansionConflict);
    }

    const auto snapshot = Instrumentation::instance().snapshot();
    const auto it = snapshot.find(InstrumentationMetric::ExpansionConflict);
    ASSERT_NE(it, snapshot.end());
    EXPECT_EQ(it->second.call_count, 5u);
    EXPECT_EQ(it->second.total_elapsed_ns, 0u);
}

TEST(InstrumentationMetricsTest, TracksBusyEdgeMasking) {
    InstrumentationGuard guard;

    // Simulate 10 busy-edge maskings
    for (int i = 0; i < 10; ++i) {
        Instrumentation::instance().increment_counter(InstrumentationMetric::BusyEdgeMasked);
    }

    const auto snapshot = Instrumentation::instance().snapshot();
    const auto it = snapshot.find(InstrumentationMetric::BusyEdgeMasked);
    ASSERT_NE(it, snapshot.end());
    EXPECT_EQ(it->second.call_count, 10u);
}

TEST(InstrumentationMetricsTest, TracksUniqueBatchPositions) {
    InstrumentationGuard guard;

    // Simulate batch collections with varying unique counts
    Instrumentation::instance().increment_counter(InstrumentationMetric::UniqueBatchPositions, 32);
    Instrumentation::instance().increment_counter(InstrumentationMetric::UniqueBatchPositions, 45);
    Instrumentation::instance().increment_counter(InstrumentationMetric::UniqueBatchPositions, 50);

    const auto snapshot = Instrumentation::instance().snapshot();
    const auto it = snapshot.find(InstrumentationMetric::UniqueBatchPositions);
    ASSERT_NE(it, snapshot.end());

    // Total unique positions across all batches
    EXPECT_EQ(it->second.call_count, 32u + 45u + 50u);

    // Can calculate average batch diversity
    double avg_unique = static_cast<double>(it->second.call_count) / 3.0;
    EXPECT_NEAR(avg_unique, 42.33, 0.1);
}

TEST(InstrumentationMetricsTest, TracksSelectionRetries) {
    InstrumentationGuard guard;

    // Simulate selection retries due to conflicts
    for (int i = 0; i < 7; ++i) {
        Instrumentation::instance().increment_counter(InstrumentationMetric::SelectionRetry);
    }

    const auto snapshot = Instrumentation::instance().snapshot();
    const auto it = snapshot.find(InstrumentationMetric::SelectionRetry);
    ASSERT_NE(it, snapshot.end());
    EXPECT_EQ(it->second.call_count, 7u);
}

TEST(InstrumentationMetricsTest, MetricToStringCoversAllMetrics) {
    // Verify all collision metrics have proper string representations
    EXPECT_EQ(metric_to_string(InstrumentationMetric::ExpansionConflict), "expansion_conflict");
    EXPECT_EQ(metric_to_string(InstrumentationMetric::BusyEdgeMasked), "busy_edge_masked");
    EXPECT_EQ(metric_to_string(InstrumentationMetric::UniqueBatchPositions), "unique_batch_positions");
    EXPECT_EQ(metric_to_string(InstrumentationMetric::SelectionRetry), "selection_retry");
}

}  // namespace
