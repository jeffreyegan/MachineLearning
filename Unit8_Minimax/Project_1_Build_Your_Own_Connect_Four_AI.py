'''
Build Your Own Connect Four AI

Now that you've gotten a sense of how the minimax algorithm works, it's time to try to create an unstoppable Connect Four AI. To create the smartest AI possible, you'll write your own evaluation function. Challenge the AI that we've written to see if you can become the Codecademy Connect Four champion!

If you get stuck during this project, check out the project walkthrough video which can be found at the bottom of the page after the final step of the project.
Mark the tasks as complete by checking them off
Changing Strategies
1.

We've imported the Connect Four game engine along with the completed minimax() function that we wrote during the lesson — including alpha-beta pruning.

However, notice that we added a new parameter to the minimax() function. The last parameter now represents the name of the evaluation function that you want to run. This essentially lets you swap out the "strategy" of your AI.

Right now in two_ai_game() player "X" is using the evaluation function codecademy_evaluate_board. This is our secret evaluation function that you're going to try to beat.

For right now, let's finish the two_ai_game() function. The second player needs to know which evaluation function to use. Right now, we only have one — codecademy_evaluate_board.

Fill that in as the last parameter of the second minimax call and run your code to see two AIs with the same strategy play each other. Remember, changing the third parameter will change how far down the game tree the algorithm looks. Changing this parameter will change the "intelligence" of the AI.
2.

Let's prove to ourselves we can easily change an AI's strategy. Write a function called random_eval() that takes board as a parameter.

random_eval() isn't going to use any strategy. It's just going to return a random number between -100 and 100. You can get a random number between -100 and 100 by using random.randint(-100, 100).

Remember, a good evaluation function should return a large positive number if it looks like "X" is winning and a large negative number if "O" is winning. This evaluation function will be pretty useless!
3.

Let's replace the "X" player's strategy with this new random one. Change the last parameter in "X"'s minimax call to random_eval.

Which player do you expect to win now?
Make Your Own Strategy
4.

It's now time to write your own strategy for your AI. We'll help you get started, but we want to see what you can come up with. To begin, create a function named my_evaluate_board() that takes board as a parameter.
5.

Let's first see if either player has won the game. You can use the has_won() function to check to see if a player has won.

has_won() takes two parameters — board and "X" or "O".

If "X" has won the game, return float("Inf"). If "O" has won the game, return -float("Inf").
6.

Now we need to figure out how to evaluate a board if neither player has won. This is where things get a little trickier.

If the game isn't over, a good strategy would be to have more streaks of two or streaks of three than your opponent. Having these streaks means that you're closer to getting a streak of four and winning the game!

One of the trickiest part of counting streaks is that they can happen in eight different directions — up, up-right, right, down-right, down, down-left, left, and up-left.

For now, let's just keep track of streaks to the right.

Inside your function after the if statements, create two variables named x_two_streak and o_two_streak and start them at 0.
7.

Now we want to loop through every space on the board and see if there's the same symbol to the right. First, let's set up a loop that goes through every piece:

for col in range(len(board)):
  for row in range(len(board[0])):
    # Check the value of board[col][row]

As this loop runs, we'll look through each piece of the board starting at the top of the left-most column. The loop will go down that column until it reaches the bottom of the board and then jumps to the top of the second column.

If board[col][row] and board[col + 1][row] are both "X", then we've found a streak of two going to the right. You should increase x_two_streak by one.

Do the same for o_two_streak.

One thing to note is that we don't want to check the final column, because that column doesn't have a column to the right. So the outer for loop should actually look like this: for col in range(len(board) -1).
8.

Finally, after finding the "X" streaks and the "O" streaks, what should we do with them? Well, if "X" has more streaks, that means they are winning and we should return a positive number. If "O" has more streaks, we should return a negative number.

Returning x_two_streak - o_two_streak will do exactly this!
Testing Our Evaluation Function
9.

Now that we've written this evaluation function, let's test it to make sure it's working correctly. To begin, comment out the function call of two_ai_game() at the bottom of your code. We don't want to play a full game until we know our evaluation function is working correctly.

Next, create a new board named new_board by calling the make_board() function. You can see we do this at the top of two_ai_game().
10.

Make a few moves on the board. You can do this using the select_space() function. The following code would put an "X" in column 6:

 select_space(new_board, 6, "X")

You can print out the board using print_board(new_board). Make enough moves so there are a couple of "X" two streaks and a couple "O" two streaks.
11.

Print the result of my_evaluate_board(new_board). Is it what you expected?

You might want to also set up the board so "X" or "O" wins to make sure those conditions are working correctly too.
12.

Assuming your function is working correctly, let's plug this strategy into our two_ai_game() function.

Find the call of minimax for the "X" player and make the last parameter my_evaluate_board.

Uncomment your call to two_ai_game() and watch your AI play ours! Feel free to adjust that third parameter to make either AI more or less "intelligent".
Expand Your Evaluation Function
13.

You have a good foundation for your evaluation function, but there's so much more you can do! Here are some ideas on how to expand your function:

    Check for streaks in all eight directions.
    Weight streaks of three higher than streaks of two.
    Only count streaks if your opponent hasn't blocked the possibility of a Connect Four. For example, if there's an "X" streak of two to the right, but the next column over has an "O", then that streak is useless.
    Only count streaks if there's enough board space to make a Connect Four. For example, there's no need to check for left streaks of two in the second column from the left. Even if there is a streak of two, you can't make a Connect Four going to the left, so the streak is useless.

Testing your evaluation function on test boards is critically important before plugging it into the two_ai_game() function. Make sure you know that your function is working how you expect it to before challenging our AI.

In the hint, we'll show you the code for our evaluation function. We strongly recommend trying to create your own function before looking at ours!


14.

If you are stuck on the project or would like to see an experienced developer work through the project, watch the following project walkthrough video!

https://youtu.be/veTCCZIvQ8c
'''

