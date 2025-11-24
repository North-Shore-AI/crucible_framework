defprotocol Crucible.Protocols.DeepJason do
  @doc """
  Helper protocol to convert nested structs into Jason-friendly maps.
  """
  def to_map(term)
end

defimpl Crucible.Protocols.DeepJason, for: Map do
  def to_map(map) do
    map
    |> Enum.map(fn {k, v} -> {k, Crucible.Protocols.DeepJason.to_map(v)} end)
    |> Enum.into(%{})
  end
end

defimpl Crucible.Protocols.DeepJason, for: List do
  def to_map(list), do: Enum.map(list, &Crucible.Protocols.DeepJason.to_map/1)
end

defimpl Crucible.Protocols.DeepJason, for: Any do
  def to_map(term), do: term
end
