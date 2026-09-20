%{
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "especifica_7.h"
#include "especifica_7.tab.h"
%}

%option noyywrap

%%

[ \t\r\n]+               { }
"#" [^\n]*              { }

":"                     { return ':'; }
"["                     { return '['; }
"]"                     { return ']'; }
","                     { return ','; }

[-+]?[0-9]+             { 
                          yylval.node = create_int_node(atoi(yytext)); 
                          return VALOR_INT; 
                        }

[-+]?[0-9]+\.[0-9]+     { 
                          yylval.node = create_float_node(atof(yytext)); 
                          return VALOR_FLOAT; 
                        }

\"([^\"\n])*\"|“([^”\n])*” { 
                          int len = strlen(yytext);
                          char *clean_str = strndup(yytext + 1, len - 2);
                          yylval.node = create_string_node(clean_str);
                          free(clean_str);
                          return VALOR_STRING; 
                        }

[a-zA-Z_][a-zA-Z0-9_]*  { 
                          yylval.str = strdup(yytext); 
                          return CHAVE; 
                        }

.                       { }

%%