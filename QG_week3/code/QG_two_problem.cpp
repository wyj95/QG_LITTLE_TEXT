#include<sort.h>
/**
 *  @name        : void swap(int *a,int *b)
 *  @description : ����ab��ֵ 
 *  @param       : ��������ָ��a��b 
 */
void swap(int *a, int *b)
{
	int c;
	c = *a;
	*a = *b;
	*b = c;
}

/**
 *  @name        : void ColorSortMain()
 *  @description : ��ɫ���������� 
 *  @param       : ����ָ��a��ֻ��0��1��2Ԫ�أ������鳤��size
 */
void ColourSortMain()
{
	long int length;
	printf("Please give me the length of the list:\n");
	GetLongInt(&length); 
	int *p=(int*)malloc(sizeof(int) * length);
	if(p == NULL)
	{
		printf("The room is full!");
		exit(0);
	}
	for(long int i=0; i<length; i++)
	{
		p[i] = rand() % 3;
	}
	printf("Before sorting,they are:\n");
	Sleep(3000);
	for(long int i=0; i<length; i++)
		printf("%d", p[i]);
	printf("\n");
	printf("Give me anything,I will sort it...");
	while(getchar() != '\n');
	ColourSort(p, length);
	printf("After sorting,they are:\n");
	Sleep(3000);
	for(long i=0; i<length; i++)
	{
		printf("%d", p[i]);
	}
	printf("\n");
	free(p);
}

/**
 *  @name        : void ColorSort(int *a,long int size)
 *  @description : ��ɫ����
 *  @param       : ����ָ��a��ֻ��0��1��2Ԫ�أ������鳤��size
 */
void ColourSort(int *a,long int size)
{
	int *pbegin=a, *pend=a+size-1;
	long int i=-1;
	while(a+i++ < pend)
	{
		if(a[i] == 2)
		{
			a[i] = 1;
			*(pbegin++) = 2;
		}
		else if(a[i] == 0)
		{
			a[i] = 1;
			*(pend--) = 0;
		}
		/* �˴� a[i]=1 Ҳ���Ե�ѭ���������pbegin��pend���θ�ֵ1 */ 
	}
}

/**
 *  @name        : int FindKthMain()
 *  @description : ���������� 
 *  @param       : None
 */
void FindKthMain()
{
	long int length, flag;
	printf("Please give me the length of the list:\n");
	GetLongInt(&length); 
	int *p=(int*)malloc(sizeof(int) * length);
	if(p == NULL)
	{
		printf("The room is full!");
		exit(0);
	}
	random_number(p, length);
	printf("Please give me the number th after sorting:\n");
	while(1)
	{
		flag = 0;
		GetLongInt(&flag);
		if(flag <= length)
			break;
		else
			printf("Please give me a number which is smaller than %ld", length);
	}
	printf("The %ld th number is %d\n", flag, FindKth(p, flag-1, 0, length-1));
}

/**
 *  @name        : int FindKth(int *a, int key, int begin, int end)
 *  @description : ��һ�������������ҵ���key��/С����
 *  @param       : ����ָ��a��Ŀ�����key��key���ڵ�����(begin,end) Ҳ������������Χ 
 */
int FindKth(int *a, long int key, long int begin, long int end)
{
	long int i=begin, j=end;
	int t=a[i];
	while(i < j)
	{
		while(a[j]<t && j>i)
			j--;
		if(j > i)
			a[i++]=a[j];
		while(a[i]>t && j>i)
			i++;
		if(j > i)
			a[j--]=a[i];
	}
	a[i]=t;
	if(i>key)
		return FindKth(a, key, begin, i);
	else if(i<key)
		return FindKth(a, key, i+1, end);
	else
		return a[i];
	/* ���ݲ�ͬȡֵ���������������С���䣬�������keyһ�������鷶Χ�ڣ��ʲ������end<begin��� */ 
}


/**
 *  @name        : void random_number(int *p,long int size)
 *  @description : �������size�����ִ洢��p��
 *  @param       : ����ָ��p�������Сsize 
 */
void random_number(int *p,long int size)
{
	for(long int i=0; i<size; i++)
		p[i] = rand();
}
