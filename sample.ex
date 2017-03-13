defp asdf(arg) do
	1 + 2
	1+3
	
	2+ 4





    # comment1
	# comment 2





	# comment3

	2 + 5
end

defp asdf2(arg) do




	1 + 3


end


defp asdf2(arg) do
	1 + 3
end

def asdf(arg) do
	1 + 2 + 3
end
module = env.module(arg)

with {:ok, date} <- Calendar.ISO.date(year, month, day),
     {:ok, time} <- Time.new(hour, minute, second, microsecond),
     do: new(date, time)

defmacro system_env(name, alt) do
  env_name = Atom.to_string(name) |> String.upcase    
    quote do      
      def unquote(name)() do
        System.get_env(unquote(env_name)) || unquote(alt)
        System.get_env(unquote(env_name)) || unquote(alt)
      end
    end
end
