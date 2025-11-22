defmodule Crucible.Ensemble.MLVotingTest do
  use ExUnit.Case, async: true

  alias Crucible.Ensemble.MLVoting

  describe "vote/3" do
    test "majority vote selects most common response" do
      responses = [
        {%{name: "m1", weight: 0.3}, {:ok, "answer_a"}, 100},
        {%{name: "m2", weight: 0.3}, {:ok, "answer_a"}, 120},
        {%{name: "m3", weight: 0.4}, {:ok, "answer_b"}, 110}
      ]

      {:ok, result} = MLVoting.vote(responses, :majority, [])
      assert result == "answer_a"
    end

    test "weighted vote uses adapter weights" do
      responses = [
        {%{name: "m1", weight: 0.6}, {:ok, "answer_a"}, 100},
        {%{name: "m2", weight: 0.3}, {:ok, "answer_b"}, 120},
        {%{name: "m3", weight: 0.1}, {:ok, "answer_b"}, 110}
      ]

      {:ok, result} = MLVoting.vote(responses, :weighted, [])
      assert result == "answer_a"
    end

    test "best_confidence selects highest confidence" do
      responses = [
        {%{name: "m1", weight: 0.3}, {:ok, %{text: "answer_a", confidence: 0.7}}, 100},
        {%{name: "m2", weight: 0.3}, {:ok, %{text: "answer_b", confidence: 0.9}}, 120},
        {%{name: "m3", weight: 0.4}, {:ok, %{text: "answer_c", confidence: 0.5}}, 110}
      ]

      {:ok, result} = MLVoting.vote(responses, :best_confidence, [])
      assert result.text == "answer_b"
      assert result.confidence == 0.9
    end

    test "unanimous returns error when no consensus" do
      responses = [
        {%{name: "m1", weight: 0.3}, {:ok, "answer_a"}, 100},
        {%{name: "m2", weight: 0.3}, {:ok, "answer_b"}, 120},
        {%{name: "m3", weight: 0.4}, {:ok, "answer_a"}, 110}
      ]

      assert {:error, :no_consensus} = MLVoting.vote(responses, :unanimous, [])
    end

    test "unanimous returns result when all agree" do
      responses = [
        {%{name: "m1", weight: 0.3}, {:ok, "answer_a"}, 100},
        {%{name: "m2", weight: 0.3}, {:ok, "answer_a"}, 120},
        {%{name: "m3", weight: 0.4}, {:ok, "answer_a"}, 110}
      ]

      {:ok, result} = MLVoting.vote(responses, :unanimous, [])
      assert result == "answer_a"
    end

    test "custom vote uses provided function" do
      responses = [
        {%{name: "m1", weight: 0.3}, {:ok, "answer_a"}, 100},
        {%{name: "m2", weight: 0.3}, {:ok, "answer_b"}, 120},
        {%{name: "m3", weight: 0.4}, {:ok, "answer_c"}, 50}
      ]

      # Custom vote: select fastest response
      vote_fn = fn resps ->
        {_adapter, {:ok, result}, _latency} = Enum.min_by(resps, fn {_, _, lat} -> lat end)
        {:ok, result}
      end

      {:ok, result} = MLVoting.vote(responses, :custom, vote_fn: vote_fn)
      assert result == "answer_c"
    end
  end

  describe "extract_confidence/1" do
    test "extracts confidence from map with confidence key" do
      result = %{confidence: 0.85, text: "answer"}
      assert MLVoting.extract_confidence(result) == 0.85
    end

    test "computes confidence from logprobs" do
      # Log probabilities that should average to ~0.74
      logprobs = [-0.3, -0.3, -0.3]
      result = %{logprobs: logprobs, text: "answer"}
      confidence = MLVoting.extract_confidence(result)
      assert confidence > 0.7 and confidence < 0.8
    end

    test "returns default confidence when no data available" do
      result = %{text: "answer"}
      assert MLVoting.extract_confidence(result) == 0.5
    end
  end
end
