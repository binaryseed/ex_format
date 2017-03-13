defmodule Formatter.CLI do


  def main(args) do
    # comment1
    # comment2
    {opts,_,_}= OptionParser.parse(args,switches: [file: :string],aliases: [f: :file])
    Formatter.format(opts[:file]) #comment3
  end
end

