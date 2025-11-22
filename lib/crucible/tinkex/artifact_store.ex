defmodule Crucible.Tinkex.ArtifactStore do
  @moduledoc """
  Local caching and storage for checkpoint artifacts.

  Provides a file-based cache for downloaded checkpoints with checksum
  verification, size tracking, and automatic cleanup based on storage limits.

  ## Configuration

      config :crucible, :artifact_cache_dir, "~/.cache/crucible/artifacts"

  ## Examples

      # Store an artifact
      {:ok, artifact} = ArtifactStore.store("model-weights", binary_content)

      # Retrieve artifact
      {:ok, artifact} = ArtifactStore.get("model-weights")

      # Verify integrity
      :ok = ArtifactStore.verify_checksum(artifact.local_path, artifact.checksum)

      # Cleanup old artifacts
      {:ok, removed} = ArtifactStore.cleanup(1_000_000_000)  # 1GB limit
  """

  require Logger

  @type artifact :: %{
          name: String.t(),
          local_path: String.t(),
          remote_path: String.t() | nil,
          size_bytes: non_neg_integer(),
          checksum: String.t(),
          downloaded_at: DateTime.t()
        }

  # ETS table for artifact metadata
  @table_name :crucible_artifact_store

  @doc """
  Returns the configured cache directory.
  """
  @spec cache_dir() :: String.t()
  def cache_dir do
    Application.get_env(:crucible, :artifact_cache_dir, default_cache_dir())
  end

  @doc """
  Stores content as an artifact with checksum.

  ## Options
    * `:remote_path` - Original remote path (for reference)
  """
  @spec store(String.t(), binary(), keyword()) ::
          {:ok, artifact()} | {:error, term()}
  def store(name, content, opts \\ []) do
    ensure_initialized()
    ensure_cache_dir()

    checksum = compute_checksum(content)
    local_path = artifact_path(name)

    # Ensure parent directory exists
    File.mkdir_p!(Path.dirname(local_path))

    case File.write(local_path, content) do
      :ok ->
        artifact = %{
          name: name,
          local_path: local_path,
          remote_path: Keyword.get(opts, :remote_path),
          size_bytes: byte_size(content),
          checksum: checksum,
          downloaded_at: DateTime.utc_now()
        }

        :ets.insert(@table_name, {name, artifact})

        emit_telemetry(:artifact_stored, artifact)

        {:ok, artifact}

      {:error, reason} ->
        {:error, {:write_failed, reason}}
    end
  end

  @doc """
  Retrieves artifact metadata by name.
  """
  @spec get(String.t()) :: {:ok, artifact()} | {:error, :not_found}
  def get(name) do
    ensure_initialized()

    case :ets.lookup(@table_name, name) do
      [{^name, artifact}] -> {:ok, artifact}
      [] -> {:error, :not_found}
    end
  end

  @doc """
  Checks if an artifact exists.
  """
  @spec exists?(String.t()) :: boolean()
  def exists?(name) do
    ensure_initialized()

    case :ets.lookup(@table_name, name) do
      [{^name, _}] -> true
      [] -> false
    end
  end

  @doc """
  Deletes an artifact and its file.
  """
  @spec delete(String.t()) :: :ok | {:error, :not_found}
  def delete(name) do
    ensure_initialized()

    case :ets.lookup(@table_name, name) do
      [{^name, artifact}] ->
        # Delete file
        if File.exists?(artifact.local_path) do
          File.rm!(artifact.local_path)
        end

        # Remove from ETS
        :ets.delete(@table_name, name)

        emit_telemetry(:artifact_deleted, artifact)

        :ok

      [] ->
        {:error, :not_found}
    end
  end

  @doc """
  Lists all stored artifacts.
  """
  @spec list() :: [artifact()]
  def list do
    ensure_initialized()

    @table_name
    |> :ets.tab2list()
    |> Enum.map(fn {_name, artifact} -> artifact end)
  end

  @doc """
  Returns total size of all artifacts in bytes.
  """
  @spec total_size() :: non_neg_integer()
  def total_size do
    list()
    |> Enum.reduce(0, fn artifact, acc -> acc + artifact.size_bytes end)
  end

  @doc """
  Cleans up artifacts to stay under the size limit.

  Removes oldest artifacts first until total size is under the limit.
  Returns the number of artifacts removed.
  """
  @spec cleanup(non_neg_integer()) :: {:ok, non_neg_integer()}
  def cleanup(max_size_bytes) do
    ensure_initialized()

    current_size = total_size()

    if current_size <= max_size_bytes do
      {:ok, 0}
    else
      # Sort by downloaded_at (oldest first)
      sorted =
        list()
        |> Enum.sort_by(& &1.downloaded_at, {:asc, DateTime})

      {_remaining_size, removed_count} =
        Enum.reduce_while(sorted, {current_size, 0}, fn artifact, {size, count} ->
          if size <= max_size_bytes do
            {:halt, {size, count}}
          else
            delete(artifact.name)
            {:cont, {size - artifact.size_bytes, count + 1}}
          end
        end)

      {:ok, removed_count}
    end
  end

  @doc """
  Downloads a checkpoint from Tinkex to local storage.

  ## Options
    * `:rest_client` - Tinkex RestClient to use
    * `:force` - Overwrite existing file (default: false)
    * `:progress` - Progress callback function
  """
  @spec download_checkpoint(String.t(), keyword()) ::
          {:ok, artifact()} | {:error, term()}
  def download_checkpoint(checkpoint_path, opts \\ []) do
    rest_client = Keyword.get(opts, :rest_client)

    if rest_client do
      output_dir = cache_dir()

      case Tinkex.CheckpointDownload.download(rest_client, checkpoint_path,
             output_dir: output_dir,
             force: Keyword.get(opts, :force, false),
             progress: Keyword.get(opts, :progress)
           ) do
        {:ok, result} ->
          # Read the downloaded content and store
          local_path = result.destination
          name = Path.basename(local_path)

          # Get file info for size
          %{size: size} = File.stat!(local_path)
          content = File.read!(local_path)
          checksum = compute_checksum(content)

          artifact = %{
            name: name,
            local_path: local_path,
            remote_path: checkpoint_path,
            size_bytes: size,
            checksum: checksum,
            downloaded_at: DateTime.utc_now()
          }

          ensure_initialized()
          :ets.insert(@table_name, {name, artifact})

          emit_telemetry(:checkpoint_downloaded, artifact)

          {:ok, artifact}

        {:error, reason} ->
          {:error, reason}
      end
    else
      {:error, :no_rest_client}
    end
  end

  @doc """
  Verifies a file's checksum matches the expected value.
  """
  @spec verify_checksum(String.t(), String.t()) ::
          :ok | {:error, :checksum_mismatch | :file_not_found}
  def verify_checksum(local_path, expected_checksum) do
    if File.exists?(local_path) do
      content = File.read!(local_path)
      actual = compute_checksum(content)

      if actual == expected_checksum do
        :ok
      else
        {:error, :checksum_mismatch}
      end
    else
      {:error, :file_not_found}
    end
  end

  # Private Functions

  defp ensure_initialized do
    if :ets.info(@table_name) == :undefined do
      :ets.new(@table_name, [:named_table, :public, :set])
    end
  end

  defp ensure_cache_dir do
    dir = cache_dir()

    unless File.exists?(dir) do
      File.mkdir_p!(dir)
    end
  end

  defp default_cache_dir do
    Path.join([System.user_home!(), ".cache", "crucible", "artifacts"])
  end

  defp artifact_path(name) do
    # Sanitize name for filesystem
    safe_name = String.replace(name, ~r/[^a-zA-Z0-9_.-]/, "_")
    Path.join(cache_dir(), safe_name)
  end

  defp compute_checksum(content) do
    :crypto.hash(:sha256, content)
    |> Base.encode16(case: :lower)
  end

  defp emit_telemetry(event, artifact) do
    :telemetry.execute(
      [:crucible, :tinkex, :artifact, event],
      %{
        timestamp: System.system_time(:millisecond),
        size_bytes: artifact.size_bytes
      },
      %{
        artifact_name: artifact.name,
        local_path: artifact.local_path
      }
    )
  end
end
