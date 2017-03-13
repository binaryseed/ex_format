import Kernel, except: [to_string: 1]
require IEx

defmodule Formatter do
  @typedoc "Abstract Syntax Tree (AST)"
  @type t :: expr | {t, t} | atom | number | binary | pid | fun | [t]
  @type expr :: {expr | atom, Keyword.t, atom | [t]}

  @binary_ops [:===, :!==,
    :==, :!=, :<=, :>=,
    :&&, :||, :<>, :++, :--, :\\, :::, :<-, :.., :|>, :=~,
    :<, :>, :->,
    :+, :-, :*, :/, :=, :|, :.,
    :and, :or, :when, :in,
    :~>>, :<<~, :~>, :<~, :<~>, :<|>,
    :<<<, :>>>, :|||, :&&&, :^^^, :~~~]

  @doc false
  defmacro binary_ops, do: @binary_ops

  @unary_ops [:!, :@, :^, :not, :+, :-, :~~~, :&]

  @doc false
  defmacro unary_ops, do: @unary_ops

  @spec binary_op_props(atom) :: {:left | :right, precedence :: integer}
  defp binary_op_props(o) do
    case o do
      o when o in [:<-, :\\]                  -> {:left,  40}
      :when                                   -> {:right, 50}
      :::                                     -> {:right, 60}
      :|                                      -> {:right, 70}
      :=                                      -> {:right, 90}
      o when o in [:||, :|||, :or]            -> {:left, 130}
      o when o in [:&&, :&&&, :and]           -> {:left, 140}
      o when o in [:==, :!=, :=~, :===, :!==] -> {:left, 150}
      o when o in [:<, :<=, :>=, :>]          -> {:left, 160}
      o when o in [:|>, :<<<, :>>>, :<~, :~>,
                :<<~, :~>>, :<~>, :<|>, :^^^] -> {:left, 170}
      :in                                     -> {:left, 180}
      o when o in [:++, :--, :.., :<>]        -> {:right, 200}
      o when o in [:+, :-]                    -> {:left, 210}
      o when o in [:*, :/]                    -> {:left, 220}
      :.                                      -> {:left, 310}
    end
  end

  def format(file_name) do
    file_content = File.read!(file_name)
    lines = String.split(file_content, "\n")
    Agent.start_link(fn -> %{} end, name: :lines)
    Agent.start_link(fn -> %{} end, name: :comments)
    for {s, i} <- Enum.with_index(lines) do
      s = String.trim(s)
      update_line(i+1, s)
      if String.first(s) == "#" do
        Agent.update(:comments, fn map ->
          Map.put(map, i+1, String.slice(s, 1, String.length(s)))
        end)
      end
    end
    # TODO: retrieve comments from file_content via tokenization
    ast = elem(Code.string_to_quoted(file_content), 1)
    IO.inspect ast
    IO.puts "\n"
    IO.puts to_string(ast, 0)
    # IO.puts "liness"
    # IO.inspect lines
    # IO.puts "\n"
    # IO.inspect Agent.get(:lines, fn map -> map end)
  end

  defp get_line(k), do: Agent.get(:lines, fn map -> Map.get(map, k) end)
  defp update_line(k, v), do: Agent.update(:lines, fn map -> Map.put(map, k, v) end)

  defp get_lineno({_, [line: lineno], _} = ast, prev), do: lineno
  defp get_lineno(_, prev), do: prev

  defp get_newline(curr, prev) when curr < prev, do: ""
  defp get_newline(curr, prev), do: get_newline(curr-1, prev, get_line(curr))
  defp get_newline(curr, prev, ""), do: "\n"
  defp get_newline(curr, prev, _), do: ""

  defp get_comments(curr, prev) when curr < prev, do: ""
  defp get_comments(curr, prev, _) when curr < prev, do: ""
  defp get_comments(curr, prev), do: get_comments(curr, prev, get_line(curr))
  defp get_comments(curr, prev, "#" <> s) do
    s = get_newline(curr-1, prev) <> "# " <> String.trim(s) <> "\n"
    get_comments(curr-1, prev, get_line(curr-1)) <> s
  end
  defp get_comments(curr, prev, ""), do: get_comments(curr-1, prev, get_line(curr-1))
  defp get_comments(curr, prev, _), do: ""

  @doc """
  Converts the given expression to a binary.
  The given `fun` is called for every node in the AST with two arguments: the
  AST of the node being printed and the string representation of that same
  node. The return value of this function is used as the final string
  representation for that AST node.
  ## Examples
      iex> Macro.to_string(quote(do: foo.bar(1, 2, 3)))
      "foo.bar(1, 2, 3)"
      iex> Macro.to_string(quote(do: 1 + 2), fn
      ...>   1, _string -> "one"
      ...>   2, _string -> "two"
      ...>   _ast, string -> string
      ...> end)
      "one + two"
  """
  @spec to_string(Macro.t, number) :: String.t
  @spec to_string(Macro.t, number, (Macro.t, Macro.t, String.t -> String.t)) :: String.t
  def to_string(tree, prev, fun \\ fn(ast, prev, string) ->
    curr = get_lineno(ast, prev)
    if curr - prev > 1 do
      IO.puts("LINENO DIFF, curr: #{curr}, prev: #{prev}")
      # IO.puts(":::::::::::string::::::::")
      # IO.puts("#{string}")
      # IO.puts "\n"
      # string = get_comments(curr-1, prev) <> get_newline(curr-1, prev) <> string
      string = get_comments(curr-1, prev) <> get_newline(curr-1, prev) <> string
    end
    string
  end)

  # Variables
  def to_string({var, _, atom} = ast, prev, fun) when is_atom(atom) do
    fun.(ast, prev, Atom.to_string(var))
  end

  # Aliases
  def to_string({:__aliases__, _, refs} = ast, prev, fun) do
    fun.(ast, prev, Enum.map_join(refs, ".", &call_to_string(&1, get_lineno(ast, prev), fun)))
  end

  # Blocks
  def to_string({:__block__, _, [expr]} = ast, prev, fun) do
    fun.(ast, prev, to_string(expr, get_lineno(ast, prev), fun))
  end

  def to_string({:__block__, _, _} = ast, prev, fun) do
    block = adjust_new_lines block_to_string(ast, prev, fun), "\n  "
    fun.(ast, prev, "(\n  " <> block <> "\n)")
  end

  # Bits containers
  def to_string({:<<>>, _, parts} = ast, prev, fun) do
    if interpolated?(ast) do
      fun.(ast, prev, interpolate(ast, prev, fun))
    else
      result = Enum.map_join(parts, ", ", fn(part) ->
        str = bitpart_to_string(part, get_lineno(ast, prev), fun)
        if :binary.first(str) == ?< or :binary.last(str) == ?> do
          "(" <> str <> ")"
        else
          str
        end
      end)
      fun.(ast, prev, "<<" <> result <> ">>")
    end
  end

  # Tuple containers
  def to_string({:{}, _, args} = ast, prev, fun) do
    tuple = "{" <> Enum.map_join(args, ", ", &to_string(&1, get_lineno(ast, prev), fun)) <> "}"
    fun.(ast, prev, tuple)
  end

  # Map containers
  def to_string({:%{}, _, args} = ast, prev, fun) do
    map = "%{" <> map_to_string(args, get_lineno(ast, prev), fun) <> "}"
    fun.(ast, prev, map)
  end

  def to_string({:%, _, [structname, map]} = ast, prev, fun) do
    {:%{}, _, args} = map
    struct = "%" <> to_string(structname, get_lineno(ast, prev), fun) <> "{" <>
              map_to_string(args, get_lineno(ast, prev), fun) <> "}"
    fun.(ast, prev, struct)
  end

  # Fn keyword
  def to_string({:fn, _, [{:->, _, [_, tuple]}] = arrow} = ast, prev, fun)
      when not is_tuple(tuple) or elem(tuple, 0) != :__block__ do
    fun.(ast, prev, "fn " <> arrow_to_string(arrow, get_lineno(ast, prev), fun) <> " end")
  end

  def to_string({:fn, _, [{:->, _, _}] = block} = ast, prev, fun) do
    fun.(ast, prev, "fn " <> block_to_string(block, get_lineno(ast, prev), fun) <> "\nend")
  end

  def to_string({:fn, _, block} = ast, prev, fun) do
    block = adjust_new_lines block_to_string(block, get_lineno(ast, prev), fun), "\n  "
    fun.(ast, prev, "fn\n  " <> block <> "\nend")
  end

  # Ranges
  def to_string({:.., _, args} = ast, prev, fun) do
    range = Enum.map_join(args, "..", &to_string(&1, get_lineno(ast, prev), fun))
    fun.(ast, prev, range)
  end

  # left -> right
  def to_string([{:->, _, _} | _] = ast, prev, fun) do
    fun.(ast, prev, "(" <> arrow_to_string(ast, prev, fun, true) <> ")")
  end

  # left when right
  def to_string({:when, _, [left, right]} = ast, prev, fun) do
    right =
      if right != [] and Keyword.keyword?(right) do
        kw_list_to_string(right, get_lineno(ast, prev), fun)
      else
        fun.(ast, prev, op_to_string(right, get_lineno(ast, prev), fun, :when, :right))
      end

    fun.(ast, prev, op_to_string(left, get_lineno(ast, prev), fun, :when, :left) <> " when " <> right)
  end

  # Binary ops
  def to_string({op, _, [left, right]} = ast, prev, fun) when op in unquote(@binary_ops) do
    fun.(ast, prev, op_to_string(left, get_lineno(ast, prev), fun, op, :left) <>
      " #{op} " <> op_to_string(right, get_lineno(ast, prev), fun, op, :right))
  end

  # Splat when
  def to_string({:when, _, args} = ast, prev, fun) do
    {left, right} = :elixir_utils.split_last(args)
    fun.(ast, prev, "(" <>
      Enum.map_join(left, ", ", &to_string(&1, get_lineno(ast, prev), fun)) <>
      ") when " <> to_string(right, get_lineno(ast, prev), fun))
  end

  # Capture
  def to_string({:&, _, [{:/, _, [{name, _, ctx}, arity]}]} = ast, prev, fun)
      when is_atom(name) and is_atom(ctx) and is_integer(arity) do
    fun.(ast, prev, "&" <> Atom.to_string(name) <> "/" <> to_string(arity, get_lineno(ast, prev), fun))
  end

  def to_string({:&, _, [{:/, _, [{{:., _, [mod, name]}, _, []}, arity]}]} = ast, prev, fun)
      when is_atom(name) and is_integer(arity) do
    fun.(ast, prev, "&" <> to_string(mod, fun) <> "." <>
      Atom.to_string(name) <> "/" <> to_string(arity, get_lineno(ast, prev), fun))
  end

  def to_string({:&, _, [arg]} = ast, prev, fun) when not is_integer(arg) do
    fun.(ast, prev, "&(" <> to_string(arg, get_lineno(ast, prev), fun) <> ")")
  end

  # Unary ops
  def to_string({unary, _, [{binary, _, [_, _]} = arg]} = ast, prev, fun)
      when unary in unquote(@unary_ops) and binary in unquote(@binary_ops) do
    fun.(ast, prev, Atom.to_string(unary) <> "(" <> to_string(arg, get_lineno(ast, prev), fun) <> ")")
  end

  def to_string({:not, _, [arg]} = ast, prev, fun)  do
    fun.(ast, prev, "not " <> to_string(arg, get_lineno(ast, prev), fun))
  end

  def to_string({op, _, [arg]} = ast, prev, fun) when op in unquote(@unary_ops) do
    fun.(ast, prev, Atom.to_string(op) <> to_string(arg, get_lineno(ast, prev), fun))
  end

  # Access
  def to_string({{:., _, [Access, :get]}, _, [{op, _, _} = left, right]} = ast, prev, fun)
      when op in unquote(@binary_ops) do
    fun.(ast, prev, "(" <> to_string(left, fun) <> ")" <> to_string([right], get_lineno(ast, prev), fun))
  end

  def to_string({{:., _, [Access, :get]}, _, [left, right]} = ast, prev, fun) do
    fun.(ast, prev, to_string(left, get_lineno(ast, prev), fun) <>
      to_string([right], get_lineno(ast, prev), fun))
  end

  # All other calls
  def to_string({target, _, args} = ast, prev, fun) when is_list(args) do
    # {_, context, _} = ast
    # if context != [] do
    #   IO.write "line: #{context[:line]}, target: "
    #   IO.inspect target
    # else

    # end
    if sigil = sigil_call(ast, prev, fun) do
      sigil
    else
      {list, last} = :elixir_utils.split_last(args)
      fun.(ast, prev, case kw_blocks?(last) do
        true  -> call_to_string_with_args(target, list, get_lineno(ast, prev), fun) <>
                  kw_blocks_to_string(last, get_lineno(ast, prev), fun)
        false -> call_to_string_with_args(target, args, get_lineno(ast, prev), fun)
      end)
    end
  end

  # Two-element tuples
  def to_string({left, right}, prev, fun) do
    to_string({:{}, [], [left, right]}, prev, fun)
  end

  # Lists
  def to_string(list, prev, fun) when is_list(list) do
    fun.(list, prev, cond do
      list == [] ->
        "[]"
      :io_lib.printable_list(list) ->
        IO.iodata_to_binary [?', Inspect.BitString.escape(IO.chardata_to_string(list), ?'), ?']
      Inspect.List.keyword?(list) ->
        "[" <> kw_list_to_string(list, prev, fun) <> "]"
      true ->
        "[" <> Enum.map_join(list, ", ", &to_string(&1, get_lineno(list, prev), fun)) <> "]"
    end)
  end

  # All other structures
  def to_string(other, prev, fun), do: fun.(other, prev, inspect(other, []))

  defp bitpart_to_string({:::, _, [left, right]} = ast, prev, fun) do
    result =
      op_to_string(left, get_lineno(ast, prev), fun, :::, :left) <>
      "::" <>
      bitmods_to_string(right, get_lineno(ast, prev), fun, :::, :right)
    fun.(ast, prev, result)
  end

  defp bitpart_to_string(ast, prev, fun) do
    to_string(ast, prev, fun)
  end

  defp bitmods_to_string({op, _, [left, right]} = ast, prev, fun, _, _) when op in [:*, :-] do
    result =
      bitmods_to_string(left, get_lineno(ast, prev), fun, op, :left) <>
      Atom.to_string(op) <>
      bitmods_to_string(right, get_lineno(ast, prev), fun, op, :right)
    fun.(ast, prev, result)
  end

  defp bitmods_to_string(other, prev, fun, parent_op, side) do
    op_to_string(other, prev, fun, parent_op, side)
  end

  # Block keywords
  @kw_keywords [:do, :catch, :rescue, :after, :else]

  defp kw_blocks?([{:do, _} | _] = kw) do
    Enum.all?(kw, &match?({x, _} when x in unquote(@kw_keywords), &1))
  end
  defp kw_blocks?(_), do: false

  # Check if we have an interpolated string.
  defp interpolated?({:<<>>, _, [_ | _] = parts}) do
    Enum.all?(parts, fn
      {:::, _, [{{:., _, [Kernel, :to_string]}, _, [_]},
                {:binary, _, _}]} -> true
      binary when is_binary(binary) -> true
      _ -> false
    end)
  end

  defp interpolated?(_) do
    false
  end

  defp interpolate({:<<>>, _, parts} = ast, prev, fun) do
    parts = Enum.map_join(parts, "", fn
      {:::, _, [{{:., _, [Kernel, :to_string]}, _, [arg]}, {:binary, _, _}]} ->
        "\#{" <> to_string(arg, get_lineno(ast, prev), fun) <> "}"
      binary when is_binary(binary) ->
        binary = inspect(binary, [])
        :binary.part(binary, 1, byte_size(binary) - 2)
    end)

    <<?", parts::binary, ?">>
  end

  defp module_to_string(atom, prev, _fun) when is_atom(atom), do: inspect(atom, [])
  defp module_to_string(other, prev, fun), do: call_to_string(other, prev, fun)

  defp sigil_call({func, _, [{:<<>>, _, _} = bin, args]} = ast, prev, fun) when is_atom(func) and is_list(args) do
    sigil =
      case Atom.to_string(func) do
        <<"sigil_", name>> ->
          "~" <> <<name>> <>
          interpolate(bin, get_lineno(ast, prev), fun) <>
          sigil_args(args, fun)
        _ ->
          nil
      end
    fun.(ast, prev, sigil)
  end

  defp sigil_call(_other, prev, _fun) do
    nil
  end

  defp sigil_args([], _fun),   do: ""
  defp sigil_args(args, fun), do: fun.(args, List.to_string(args))

  # @spec call_to_string(Macro.t, Macro.t,) :: String.t
  @spec call_to_string(Macro.t, Macro.t, (Macro.t, Macro.t, String.t -> String.t)) :: String.t
  # for 'def', function names etc.
  defp call_to_string(atom, prev, _fun) when is_atom(atom),
    do: Atom.to_string(atom)
  defp call_to_string({:., _, [{:&, _, [val]} = arg]} = ast, prev, fun) when not is_integer(val),
    do: "(" <> module_to_string(arg, get_lineno(ast, prev), fun) <> ")."
  defp call_to_string({:., _, [{:fn, _, _} = arg]} = ast, prev, fun),
    do: "(" <> module_to_string(arg, get_lineno(ast, prev), fun) <> ")."
  defp call_to_string({:., _, [arg]} = ast, prev, fun),
    do: module_to_string(arg, get_lineno(ast, prev), fun) <> "."
  # e.g. env.module()
  defp call_to_string({:., _, [left, right]} = ast, prev, fun) do
    module_to_string(left, get_lineno(ast, prev), fun) <> "." <> call_to_string(right, get_lineno(ast, prev), fun)
  end
  defp call_to_string(other, prev, fun),
    do: to_string(other, prev, fun)


  defp call_to_string_with_args(target, args, prev, fun) do
    # TODO: remove parens around def/defp/defmacro etc.
    need_parens = not target in [:def]
    target = call_to_string(target, prev, fun)
    args = args_to_string(args, prev, fun)
    if need_parens do
      target <> "(" <> args <> ")"
    else
      target <> " " <> args
    end
  end

  # turn (a, b, c) into strings
  # FIXME
  defp args_to_string(args, prev, fun) do
    {list, last} = :elixir_utils.split_last(args)
    if last != [] and Inspect.List.keyword?(last) do
      prefix =
        case list do
          [] -> ""
          _  -> Enum.map_join(list, ", ", &to_string(&1, prev, fun)) <> ", "
        end
      prefix <> kw_list_to_string(last, prev, fun)
    else
      Enum.map_join(args, ", ", &to_string(&1, prev, fun))
    end
  end

  defp kw_blocks_to_string(kw, prev, fun) do
    Enum.reduce(@kw_keywords, " ", fn(x, acc) ->
      case Keyword.has_key?(kw, x) do
        true  -> acc <> kw_block_to_string(x, Keyword.get(kw, x), prev, fun)
        false -> acc
      end
    end) <> "end"
  end

  # print do ... end
  defp kw_block_to_string(key, value, prev, fun) do
    # indent lines in block
    block = adjust_new_lines block_to_string(value, prev, fun), "\n  "
    # add \n and two spaces after 'do':
    # do\n
    # ..statement\n
    Atom.to_string(key) <> "\n  " <> block <> "\n"
  end

  defp block_to_string([{:->, _, _} | _] = block, prev, fun) do
    Enum.map_join(block, "\n", fn({:->, _, [left, right]}) ->
      left = comma_join_or_empty_paren(left, prev, fun, false)
      left <> "->\n  " <> adjust_new_lines block_to_string(right, prev, fun), "\n  "
    end)
  end


  defp block_to_string({:__block__, _, exprs} = ast, prev, fun) do
    # FIXME: is it prev or ast?
    # Enum.map_join(exprs, "\n", &to_string(&1, prev, fun))
    prev = get_lineno(ast, prev)
    Enum.map_join(exprs, "\n", fn(expr) ->
      expr_string = to_string(expr, prev, fun)
      prev = get_lineno(expr, prev)
      expr_string
    end)
  end

  defp block_to_string(other, prev, fun), do: to_string(other, prev, fun)

  defp map_to_string([{:|, _, [update_map, update_args]} = ast], prev, fun) do
    to_string(update_map, get_lineno(ast, prev), fun) <> " | " <> map_to_string(update_args, get_lineno(ast, prev), fun)
  end

  defp map_to_string(list, prev, fun) do
    cond do
      Inspect.List.keyword?(list) -> kw_list_to_string(list, prev, fun)
      true -> map_list_to_string(list, prev, fun)
    end
  end

  defp kw_list_to_string(list, prev, fun) do
    Enum.map_join(list, ", ", fn {key, value} ->
      atom_name = case Inspect.Atom.inspect(key) do
        ":" <> rest -> rest
        other       -> other
      end
      atom_name <> ": " <> to_string(value, prev, fun)
    end)
  end

  defp map_list_to_string(list, prev, fun) do
    Enum.map_join(list, ", ", fn {key, value} ->
      to_string(key, prev, fun) <> " => " <> to_string(value, prev, fun)
    end)
  end

  defp parenthise(expr, prev, fun) do
    "(" <> to_string(expr, prev, fun) <> ")"
  end

  defp op_to_string({op, _, [_, _]} = expr, prev, fun, parent_op, side) when op in unquote(@binary_ops) do
    {parent_assoc, parent_prec} = binary_op_props(parent_op)
    {_, prec}                   = binary_op_props(op)
    cond do
      parent_prec < prec -> to_string(expr, prev, fun)
      parent_prec > prec -> parenthise(expr, prev, fun)
      true ->
        # parent_prec == prec, so look at associativity.
        if parent_assoc == side do
          to_string(expr, prev, fun)
        else
          parenthise(expr, prev, fun)
        end
    end
  end

  defp op_to_string(expr, prev, fun, _, _), do: to_string(expr, prev, fun)

  defp arrow_to_string(pairs, prev, fun, paren \\ false) do
    Enum.map_join(pairs, "; ", fn({:->, _, [left, right]}) ->
      left = comma_join_or_empty_paren(left, prev, fun, paren)
      left <> "-> " <> to_string(right, prev, fun)
    end)
  end

  defp comma_join_or_empty_paren([], prev, _fun, true),  do: "() "
  defp comma_join_or_empty_paren([], prev, _fun, false), do: ""

  defp comma_join_or_empty_paren(left, prev, fun, _) do
    Enum.map_join(left, ", ", &to_string(&1, prev, fun)) <> " "
  end

  defp adjust_new_lines(block, replacement) do
    for <<x <- block>>, into: "" do
      case x == ?\n do
        true  -> replacement
        false -> <<x>>
      end
    end
  end
end
