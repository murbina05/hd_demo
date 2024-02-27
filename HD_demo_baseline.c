#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>


#define MAX_VAL 10000
#define MIN_VAL 0

void print_vector(int *vector, int N)
{
    for (int i=0; i<N; i++)
        printf("[%d] ", vector[i]);
    printf("\n\n");
}

void print_matrix(int *matrix, int R, int C)
{
    for (int i=0; i<R; i++)
    {
        for (int j=0; j<C; j++)
            printf("[%d] ", matrix[i*C + j]);
        printf("\n");
    }
    printf("\n\n");
}


void bind(int *inA, int *inB, int *out, int dimensionality)
{
    // printf("BIND !!!\n" );
    // print_vector(inA, dimensionality);
    // print_vector(inB, dimensionality);
    for(int j=0; j<dimensionality; j++)
        out[j] = inA[j] * inB[j];
    // print_vector(inA, dimensionality);
    // print_vector(inB, dimensionality);
    // print_vector(out, dimensionality);
    // printf("\n\n\n");

}


void bundle(int *inA, int *inB, int *out, int dimensionality)
{
    // printf("BUNDLE !!!\n" );
    // print_vector(inA, dimensionality);
    // print_vector(inB, dimensionality);
    for(int j=0; j<dimensionality; j++)
        out[j] = inA[j] + inB[j];
    // print_vector(inA, dimensionality);
    // print_vector(inB, dimensionality);
    // print_vector(out, dimensionality);
    // printf("\n\n\n");
}


void generate_id_vectors(int *id_vectors, int input_size, int dimensionality)
{
    for(int i=0; i<input_size; i++)
        for(int j=0; j<dimensionality; j++)
            id_vectors[i*dimensionality + j] = (rand()%2)*2-1;
}


void select_indexes_to_flip(int *indexes, int N, int dimensionality)
{
    bool found;
    int target_idx;
    indexes[0] = rand()%dimensionality;

    for(int i=1; i<N; i++)
    {
        while(1)
        {
            found = false;
            target_idx = rand()%dimensionality;
            for(int k=0; k<i; k++)
            {
                if(target_idx == indexes[k])
                {
                    found = true;
                    break;
                }
            }
            if(!found)
            {
                indexes[i] = target_idx;
                break;
            }
        }
    }
}


void generate_level_vectors(int *level_vectors, int num_levels, int dimensionality)
{
    int t;
    int N = dimensionality/2/num_levels;
    int indexes[N];
    for(int j=0; j<dimensionality; j++) // first level vector is random
        level_vectors[j] = (rand()%2)*2-1;

    for(int i=1; i<num_levels; i++)
    {
        for(int j=0; j<dimensionality; j++) 
            level_vectors[i*dimensionality + j] = level_vectors[(i-1)*dimensionality + j];
        select_indexes_to_flip(indexes, N, dimensionality);
        for(int j=0; j<N; j++)
            level_vectors[i*dimensionality + indexes[j]] = level_vectors[i*dimensionality + indexes[j]] * (-1);
    }
}


void id_level(int *input, int *id_vectors, int *level_vectors, int *encoded_input, int num_levels, int input_size, int dimensionality)
{
    int k;
    int *binded_hypervector = malloc(dimensionality * sizeof(int));
    for(int i=0; i<input_size; i++)
    {
        k = (int) floor( ( (float)input[i]/(MAX_VAL - MIN_VAL) )*num_levels);
        bind(id_vectors + i*dimensionality, level_vectors + k*dimensionality, binded_hypervector, dimensionality);
        bundle(binded_hypervector, encoded_input, encoded_input, dimensionality);
    }
}


void generate_projection_matrix(int *proj_matrix, int input_size, int dimensionality)
{
    for(int i=0; i<input_size; i++)
        for(int j=0; j<dimensionality; j++)
            proj_matrix[i*dimensionality + j] = (rand()%2)*2-1;
}


void random_projection(int *input, int *proj_matrix, int *encoded_input, int input_size, int dimensionality)
{
    int sum;
    for(int i=0; i<dimensionality; i++)
    {
        sum = 0;
        for(int j=0; j<input_size; j++)
            sum += input[j] * proj_matrix[j*dimensionality + i];
        encoded_input[i] = sum;
    }

}

// float compute_norm(int *v, int N)
// {
//     float norm = 0;
//     for(int i=0; i<N; i++)
//         norm = norm + v[i]*v[i];
// }

int check_similarity(int *v1, int *v2, int N)
{
    int sim = 0;
    // float norm1 = compute_norm(v1, N);
    // float norm2 = compute_norm(v2, N);
    for(int i=0; i<N; i++)
        sim = sim + v1[i]*v2[i];
    // return sim / norm1 / norm2;
    return sim;

}


void classify(int *input, int *model, int C, int dimensionality)
{
    int best_class, cur_sim, best_sim = -1;
    for(int i=0; i<C; i++)
    {
        cur_sim = check_similarity(input, model + i*dimensionality, dimensionality);
        if(cur_sim > best_sim)
        {
            best_sim = cur_sim;
            best_class = i;
        }
    }
}


int main(int argc, char *argv[])
{
    int input_size = atoi(argv[1]);
    int output_size = atoi(argv[2]);
    int dimensionality = atoi(argv[3]);
    int num_levels = atoi(argv[4]);
    int quantization = atoi(argv[5]);

    int *input = malloc(input_size * sizeof(int) );
    int *encoded_input = malloc( dimensionality * sizeof(int) );
    for (int i=0; i<dimensionality; i++)
        encoded_input[i] = 0;
    for (int i=0; i<input_size; i++)
        input[i] = i;
 
    int *id_vectors = malloc( (input_size*dimensionality) * sizeof(int) );
    int *level_vectors = malloc( (num_levels*dimensionality) * sizeof(int) );
    int *model = malloc( (output_size*dimensionality) * sizeof(int) );

    for (int i=0; i<output_size*dimensionality; i++)
        model[i] = rand()%50;
    
    clock_t start, end;
    double cpu_time_used;

    generate_id_vectors(id_vectors, input_size, dimensionality);
    generate_level_vectors(level_vectors, num_levels, dimensionality);

    
    printf("******** ID-LEVEL ENCODING ********\n");
    start = clock();
    for(int i=0; i<1000; i++)
    {
        id_level(input, id_vectors, level_vectors, encoded_input, num_levels, input_size, dimensionality);
        classify(input, model, output_size, dimensionality);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("SHOW HOW MUCH TIME DID IT TAKE: %f", cpu_time_used);

    free(input);free(encoded_input);free(id_vectors);free(level_vectors);





    // printf("input vector:\n");
    // print_vector(input, input_size);
    // printf("id hypervectors:\n");
    // print_matrix(id_vectors, input_size, dimensionality);
    // printf("level hypervectors:\n");
    // print_matrix(level_vectors, num_levels, dimensionality);
    // printf("encoded hypervector:\n");
    // print_vector(encoded_input, dimensionality);

    // printf("******** PROJECTION ENCODING ********\n");
    // int proj_matrix[input_size*dimensionality];
    // generate_projection_matrix(proj_matrix, input_size, dimensionality);
    // random_projection(input, proj_matrix, encoded_input, input_size, dimensionality);
    // printf("input vector:\n");
    // print_vector(input, input_size);
    // printf("projection matrix:\n");
    // print_matrix(proj_matrix, input_size, dimensionality);
    // printf("encoded hypervector:\n");
    // print_vector(encoded_input, dimensionality);
    
    return 1;
}



