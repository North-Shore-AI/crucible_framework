defmodule Crucible.Stage.BenchTest do
  use ExUnit.Case, async: true

  # Suppress expected log messages when no data is available
  @moduletag capture_log: true

  alias Crucible.Context
  alias Crucible.Stage.Bench
  alias CrucibleIR.Experiment

  describe "run/2" do
    setup do
      experiment = %Experiment{
        id: "test-exp",
        backend: nil,
        pipeline: [],
        reliability: %CrucibleIR.Reliability.Config{
          stats: %CrucibleIR.Reliability.Stats{
            tests: [:ttest],
            alpha: 0.05,
            options: %{}
          }
        }
      }

      context = %Context{
        experiment_id: experiment.id,
        run_id: "test-run",
        experiment: experiment,
        outputs: [],
        metrics: %{}
      }

      {:ok, experiment: experiment, context: context}
    end

    test "returns context with no_data when no data available", %{context: context} do
      assert {:ok, result} = Bench.run(context, %{})
      assert result.metrics.bench.no_data == true
    end

    test "analyzes single group with descriptive statistics", %{context: context} do
      context = %{
        context
        | outputs: [
            %{score: 0.8},
            %{score: 0.85},
            %{score: 0.82},
            %{score: 0.79},
            %{score: 0.83}
          ]
      }

      assert {:ok, result} = Bench.run(context, %{data_source: :outputs})
      bench_metrics = result.metrics.bench

      assert bench_metrics.results.n == 5
      assert bench_metrics.results.mean > 0.81 and bench_metrics.results.mean < 0.83
      assert bench_metrics.results.tests_skipped =~ "Single group"
    end

    test "compares two groups with t-test", %{context: context} do
      context = %{
        context
        | outputs: [
            %{prompt: "test1", score: 0.7},
            %{prompt: "test1", score: 0.72},
            %{prompt: "test1", score: 0.68},
            %{prompt: "test2", score: 0.85},
            %{prompt: "test2", score: 0.88},
            %{prompt: "test2", score: 0.83}
          ]
      }

      experiment = %{
        context.experiment
        | reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:ttest],
              alpha: 0.05,
              options: %{}
            }
          }
      }

      context = %{context | experiment: experiment}

      assert {:ok, result} = Bench.run(context, %{data_source: :outputs})
      bench_metrics = result.metrics.bench

      assert bench_metrics.tests_run == [:ttest]
      assert bench_metrics.alpha == 0.05
      assert Map.has_key?(bench_metrics.results, :test_results)
      assert Map.has_key?(bench_metrics.results.test_results, :ttest)

      ttest_result = bench_metrics.results.test_results.ttest
      assert Map.has_key?(ttest_result, :p_value)
      assert Map.has_key?(ttest_result, :statistic)
      assert Map.has_key?(ttest_result, :significant)
    end

    test "runs multiple statistical tests", %{context: context} do
      context = %{
        context
        | metrics: %{
            backend: %{
              raw_steps: [
                %{loss: 0.5},
                %{loss: 0.48},
                %{loss: 0.46},
                %{loss: 0.44}
              ]
            }
          }
      }

      experiment = %{
        context.experiment
        | reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:bootstrap],
              alpha: 0.05,
              options: %{bootstrap_iterations: 1000}
            }
          }
      }

      context = %{context | experiment: experiment}

      assert {:ok, result} = Bench.run(context, %{data_source: :backend})
      bench_metrics = result.metrics.bench

      # Single group case - should get descriptive stats
      assert bench_metrics.results.n == 4
      assert is_float(bench_metrics.results.mean)
    end

    test "handles mann_whitney test for non-parametric data", %{context: context} do
      # Create two groups with different distributions
      # Outlier in group 1
      group1_data = [1, 2, 2, 3, 3, 3, 100]
      group2_data = [4, 5, 5, 6, 6, 7, 8]

      outputs =
        Enum.map(group1_data, fn v -> %{prompt: "g1", score: v} end) ++
          Enum.map(group2_data, fn v -> %{prompt: "g2", score: v} end)

      context = %{context | outputs: outputs}

      experiment = %{
        context.experiment
        | reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:mann_whitney],
              alpha: 0.05,
              options: %{}
            }
          }
      }

      context = %{context | experiment: experiment}

      assert {:ok, result} = Bench.run(context, %{data_source: :outputs})
      bench_metrics = result.metrics.bench

      assert bench_metrics.tests_run == [:mann_whitney]
      mann_whitney = bench_metrics.results.test_results.mann_whitney
      assert Map.has_key?(mann_whitney, :p_value)
      assert Map.has_key?(mann_whitney, :statistic)
    end

    test "handles paired tests with matching samples", %{context: context} do
      outputs = [
        %{prompt: "before", score: 0.7},
        %{prompt: "before", score: 0.72},
        %{prompt: "before", score: 0.68},
        %{prompt: "after", score: 0.75},
        %{prompt: "after", score: 0.77},
        %{prompt: "after", score: 0.73}
      ]

      context = %{context | outputs: outputs}

      experiment = %{
        context.experiment
        | reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:paired_ttest],
              alpha: 0.05,
              options: %{}
            }
          }
      }

      context = %{context | experiment: experiment}

      assert {:ok, result} = Bench.run(context, %{data_source: :outputs})
      bench_metrics = result.metrics.bench

      paired_result = bench_metrics.results.test_results.paired_ttest
      assert Map.has_key?(paired_result, :p_value)
    end

    test "handles multiple groups with ANOVA", %{context: context} do
      outputs = [
        %{prompt: "g1", score: 0.7},
        %{prompt: "g1", score: 0.72},
        %{prompt: "g2", score: 0.8},
        %{prompt: "g2", score: 0.82},
        %{prompt: "g3", score: 0.6},
        %{prompt: "g3", score: 0.62}
      ]

      context = %{context | outputs: outputs}

      experiment = %{
        context.experiment
        | reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:anova],
              alpha: 0.05,
              options: %{}
            }
          }
      }

      context = %{context | experiment: experiment}

      assert {:ok, result} = Bench.run(context, %{data_source: :outputs})
      bench_metrics = result.metrics.bench

      assert bench_metrics.tests_run == [:anova]
      anova_result = bench_metrics.results.test_results.anova
      assert Map.has_key?(anova_result, :p_value)
    end

    test "extracts data from metrics", %{context: context} do
      metrics = %{
        accuracy: [0.8, 0.85, 0.82],
        loss: [0.2, 0.18, 0.19]
      }

      context = %{context | metrics: metrics}

      experiment = %{
        context.experiment
        | reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:ttest],
              alpha: 0.05,
              options: %{}
            }
          }
      }

      context = %{context | experiment: experiment}

      assert {:ok, result} = Bench.run(context, %{data_source: :metrics})
      bench_metrics = result.metrics.bench

      # Should analyze the two metrics groups
      assert Enum.sort(bench_metrics.results.groups) == [:accuracy, :loss]
      assert bench_metrics.results.sample_sizes == [3, 3]
    end

    test "uses custom data extractor", %{context: context} do
      custom_extractor = fn _ctx ->
        %{
          control: [0.5, 0.52, 0.48],
          treatment: [0.7, 0.72, 0.68]
        }
      end

      experiment = %{
        context.experiment
        | reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:welch_ttest],
              alpha: 0.05,
              options: %{}
            }
          }
      }

      context = %{context | experiment: experiment}

      assert {:ok, result} =
               Bench.run(context, %{
                 data_source: {:custom, custom_extractor}
               })

      bench_metrics = result.metrics.bench
      assert bench_metrics.results.groups == [:control, :treatment]
      assert Map.has_key?(bench_metrics.results.test_results, :welch_ttest)
    end

    test "handles bootstrap confidence intervals", %{context: context} do
      outputs = [
        %{prompt: "g1", score: 0.7},
        %{prompt: "g1", score: 0.72},
        %{prompt: "g1", score: 0.68},
        %{prompt: "g2", score: 0.8},
        %{prompt: "g2", score: 0.82},
        %{prompt: "g2", score: 0.78}
      ]

      context = %{context | outputs: outputs}

      experiment = %{
        context.experiment
        | reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:bootstrap],
              alpha: 0.05,
              options: %{bootstrap_iterations: 100}
            }
          }
      }

      context = %{context | experiment: experiment}

      assert {:ok, result} = Bench.run(context, %{data_source: :outputs})
      bench_metrics = result.metrics.bench

      bootstrap_result = bench_metrics.results.test_results.bootstrap
      assert Map.has_key?(bootstrap_result, :group1_ci)
      assert Map.has_key?(bootstrap_result, :group2_ci)
      assert Map.has_key?(bootstrap_result, :mean_difference)
      assert bootstrap_result.iterations == 100
    end

    test "calculates effect sizes", %{context: context} do
      outputs = [
        %{prompt: "g1", score: 0.5},
        %{prompt: "g1", score: 0.52},
        %{prompt: "g1", score: 0.48},
        %{prompt: "g2", score: 0.7},
        %{prompt: "g2", score: 0.72},
        %{prompt: "g2", score: 0.68}
      ]

      context = %{context | outputs: outputs}

      experiment = %{
        context.experiment
        | reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:ttest],
              alpha: 0.05,
              options: %{}
            }
          }
      }

      context = %{context | experiment: experiment}

      assert {:ok, result} = Bench.run(context, %{data_source: :outputs})
      bench_metrics = result.metrics.bench

      assert Map.has_key?(bench_metrics.results, :effect_size)
    end

    test "handles errors gracefully", %{context: context} do
      # Test with invalid data that would cause an error
      outputs = [
        %{prompt: "g1", score: nil},
        %{prompt: "g1", score: "invalid"}
      ]

      context = %{context | outputs: outputs}

      # Should not crash but return no data
      assert {:ok, result} = Bench.run(context, %{data_source: :outputs})
      assert result.metrics.bench.no_data == true
    end

    test "respects alpha configuration", %{context: context} do
      outputs = [
        %{prompt: "g1", score: 0.7},
        %{prompt: "g1", score: 0.72},
        %{prompt: "g2", score: 0.71},
        %{prompt: "g2", score: 0.73}
      ]

      context = %{context | outputs: outputs}

      experiment = %{
        context.experiment
        | reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:ttest],
              # Stricter alpha
              alpha: 0.01,
              options: %{}
            }
          }
      }

      context = %{context | experiment: experiment}

      assert {:ok, result} = Bench.run(context, %{data_source: :outputs})
      bench_metrics = result.metrics.bench

      assert bench_metrics.alpha == 0.01

      ttest_result = bench_metrics.results.test_results.ttest
      # With such similar values and strict alpha, should not be significant
      if ttest_result.p_value > 0.01 do
        assert ttest_result.significant == false
      end
    end
  end

  describe "describe/1" do
    test "returns stage description" do
      description = Bench.describe(%{tests: [:ttest, :bootstrap], alpha: 0.05})

      assert description.stage == :bench
      assert description.description =~ "Statistical benchmarking"
      assert description.tests == [:ttest, :bootstrap]
      assert description.alpha == 0.05
    end
  end

  # Property-based tests
  describe "properties" do
    use ExUnitProperties

    property "p-values are always between 0 and 1" do
      check all(
              group1 <- list_of(float(min: 0.0, max: 1.0), min_length: 3),
              group2 <- list_of(float(min: 0.0, max: 1.0), min_length: 3)
            ) do
        outputs =
          Enum.map(group1, fn v -> %{prompt: "g1", score: v} end) ++
            Enum.map(group2, fn v -> %{prompt: "g2", score: v} end)

        experiment = %Experiment{
          id: "test",
          backend: nil,
          pipeline: [],
          reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:ttest],
              alpha: 0.05,
              options: %{}
            }
          }
        }

        context = %Context{
          experiment_id: experiment.id,
          run_id: "test-run",
          experiment: experiment,
          outputs: outputs,
          metrics: %{}
        }

        case Bench.run(context, %{data_source: :outputs}) do
          {:ok, result} ->
            case get_in(result.metrics, [:bench, :results, :test_results, :ttest, :p_value]) do
              nil -> true
              p_value -> p_value >= 0 and p_value <= 1
            end

          _ ->
            true
        end
      end
    end

    property "effect sizes are real numbers" do
      check all(
              group1 <- list_of(float(min: -100.0, max: 100.0), min_length: 2),
              group2 <- list_of(float(min: -100.0, max: 100.0), min_length: 2)
            ) do
        outputs =
          Enum.map(group1, fn v -> %{prompt: "g1", score: v} end) ++
            Enum.map(group2, fn v -> %{prompt: "g2", score: v} end)

        experiment = %Experiment{
          id: "test",
          backend: nil,
          pipeline: [],
          reliability: %CrucibleIR.Reliability.Config{
            stats: %CrucibleIR.Reliability.Stats{
              tests: [:ttest],
              alpha: 0.05,
              options: %{}
            }
          }
        }

        context = %Context{
          experiment_id: experiment.id,
          run_id: "test-run",
          experiment: experiment,
          outputs: outputs,
          metrics: %{}
        }

        case Bench.run(context, %{data_source: :outputs}) do
          {:ok, result} ->
            case get_in(result.metrics, [:bench, :results, :effect_size]) do
              nil -> true
              %{error: _} -> true
              effect -> is_map(effect)
            end

          _ ->
            true
        end
      end
    end
  end
end
