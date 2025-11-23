defmodule Crucible.Thinker.Validation.SchemaTest do
  use ExUnit.Case, async: true

  alias Crucible.Thinker.Validation.Schema

  describe "parse/1" do
    test "parses single valid claim" do
      output = "CLAIM[c1]: The study found significant results (citing 12345)"

      assert [claim] = Schema.parse(output)
      assert claim.index == 1
      assert claim.text == "The study found significant results"
      assert claim.doc_id == 12345
    end

    test "parses multiple claims" do
      output = """
      CLAIM[c1]: First claim text (citing 100)
      CLAIM[c2]: Second claim text (citing 200)
      CLAIM[c3]: Third claim text (citing 300)
      """

      claims = Schema.parse(output)
      assert length(claims) == 3
      assert Enum.map(claims, & &1.index) == [1, 2, 3]
      assert Enum.map(claims, & &1.doc_id) == [100, 200, 300]
    end

    test "returns empty list for invalid format" do
      output = "This is just plain text without claims"
      assert Schema.parse(output) == []
    end

    test "handles claims with extra whitespace" do
      output = "CLAIM[c1]:   Claim with spaces   (citing 999)"

      assert [claim] = Schema.parse(output)
      assert claim.text == "Claim with spaces"
    end

    test "handles multiline claim text" do
      output =
        "CLAIM[c1]: This is a longer claim that spans multiple words and contains detailed information (citing 555)"

      assert [claim] = Schema.parse(output)
      assert String.contains?(claim.text, "longer claim")
    end
  end

  describe "validate/1" do
    test "returns true for valid claim" do
      claim = %{index: 1, text: "Valid claim text", doc_id: 123}
      assert Schema.validate(claim) == true
    end

    test "returns false for zero index" do
      claim = %{index: 0, text: "Valid text", doc_id: 123}
      assert Schema.validate(claim) == false
    end

    test "returns false for empty text" do
      claim = %{index: 1, text: "", doc_id: 123}
      assert Schema.validate(claim) == false
    end

    test "returns false for zero doc_id" do
      claim = %{index: 1, text: "Valid text", doc_id: 0}
      assert Schema.validate(claim) == false
    end

    test "returns false for negative values" do
      claim = %{index: -1, text: "Text", doc_id: 123}
      assert Schema.validate(claim) == false
    end
  end

  describe "format/1" do
    test "formats claim back to standard format" do
      claim = %{index: 1, text: "Test claim", doc_id: 456}

      assert Schema.format(claim) == "CLAIM[c1]: Test claim (citing 456)"
    end
  end
end
